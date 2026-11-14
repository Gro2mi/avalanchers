//! Logging setup shared by the command line tool, the Python module, the web
//! build and the examples.
//!
//! Messages are written with the [`tracing`] macros. The level decides who
//! sees a message:
//!
//! | level   | use for                                                                  |
//! |---------|--------------------------------------------------------------------------|
//! | `error` | an operation failed and its result is lost or unusable                   |
//! | `warn`  | the run continues, but its result may differ from what the user expects; |
//! |         | the message says what to change                                          |
//! | `info`  | milestones of a run: GPU in use, data loaded, release area, run finished, |
//! |         | stage timings                                                            |
//! | `debug` | details for diagnosing a run: stage timings, buffer sizes, settings      |
//! | `trace` | per step, per batch or per tile details                                  |
//!
//! Messages start with a capital letter, have no trailing period, give values
//! with their unit and name settings as they are written in the settings
//! file. A second clause after a semicolon says what happens next or what to
//! change.
//!
//! The level applies to the avalanchers crates; other crates (wgpu, naga,
//! HTTP clients) only report warnings and errors unless the
//! `AVALANCHERS_LOG` environment variable sets them explicitly. The default
//! level is `warn` in release builds and `info` in debug builds. It is
//! chosen, from highest to lowest priority, by
//!
//! 1. [`set_log_level`] (the `--log-level` option of the command line tool,
//!    `avalanchers.set_log_level()` in Python, `set_log_level()` in the
//!    browser),
//! 2. the `AVALANCHERS_LOG` environment variable (falling back to
//!    `RUST_LOG`): a level such as `debug`, or full filter directives such
//!    as `info,compute_core=trace,wgpu_hal=info`,
//! 3. the default of [`init_logging_with_default`] or the build default.
//!
//! Native builds write to stderr (coloured when stderr is a terminal and
//! `NO_COLOR` is unset) or to the [`set_log_output`] callback (the Python
//! module writes to `sys.stderr`, which notebooks show), the web build to the
//! browser console. A
//! [`set_message_listener`] callback additionally receives the warnings and
//! errors, e.g. to show them in a user interface.

use std::fmt;
use std::str::FromStr;
use std::sync::{OnceLock, RwLock};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::{EnvFilter, Registry, prelude::*, reload};

/// Environment variable with the log level or filter directives.
pub const LOG_ENV_VAR: &str = "AVALANCHERS_LOG";

/// Crates whose messages follow the configured level; all other crates
/// report warnings and errors only.
const OWN_TARGETS: &[&str] = &[
    "simulation",
    "compute_core",
    "data_processor",
    "render_core",
    "cli",
    // library names of the Python and web bindings
    "_avalanchers",
    "avalanchers",
    // render_core examples
    "live_sim",
    "dem_viewer",
    "dem_snapshot",
    "video_export",
];

/// Severity threshold of the log output, from no output to everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    /// All levels, from no output to everything.
    pub const ALL: [LogLevel; 6] = [
        LogLevel::Off,
        LogLevel::Error,
        LogLevel::Warn,
        LogLevel::Info,
        LogLevel::Debug,
        LogLevel::Trace,
    ];

    /// The level of this build: `warn` in release, `info` in debug builds.
    pub const fn build_default() -> LogLevel {
        if cfg!(debug_assertions) {
            LogLevel::Info
        } else {
            LogLevel::Warn
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            LogLevel::Off => "off",
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }

    /// Filter directives: this level for the avalanchers crates, at most
    /// `warn` for all others.
    fn directives(self) -> String {
        if self <= LogLevel::Warn {
            return self.as_str().to_string();
        }
        let mut directives = String::from("warn");
        for target in OWN_TARGETS {
            directives.push_str(&format!(",{target}={}", self.as_str()));
        }
        directives
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(LogLevel::Off),
            "error" => Ok(LogLevel::Error),
            "warn" | "warning" => Ok(LogLevel::Warn),
            "info" => Ok(LogLevel::Info),
            "debug" => Ok(LogLevel::Debug),
            "trace" => Ok(LogLevel::Trace),
            other => Err(format!(
                "unknown log level '{other}', expected one of off, error, warn, info, debug, trace"
            )),
        }
    }
}

impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Off => LevelFilter::OFF,
            LogLevel::Error => LevelFilter::ERROR,
            LogLevel::Warn => LevelFilter::WARN,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Debug => LevelFilter::DEBUG,
            LogLevel::Trace => LevelFilter::TRACE,
        }
    }
}

/// Handle to swap the filter of the installed subscriber at run time; set
/// once by the first initialization.
static FILTER: OnceLock<reload::Handle<EnvFilter, Registry>> = OnceLock::new();

/// Receiver of the warnings and errors that pass the log level, called with
/// the level and the message text.
pub type MessageListener = Box<dyn Fn(LogLevel, &str) + Send + Sync>;

/// Receiver of the formatted log lines (with line break) in place of stderr.
pub type LogOutput = Box<dyn Fn(&str) + Send + Sync>;

// the web build writes to the browser console only
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct Output {
    write: LogOutput,
    ansi: bool,
}

static OUTPUT: RwLock<Option<Output>> = RwLock::new(None);

/// Sends the native log output to `output` instead of stderr, with terminal
/// colours if `ansi`; `None` restores stderr. Call it before logging is
/// initialized: the colour choice is fixed then.
pub fn set_log_output(output: Option<LogOutput>, ansi: bool) {
    if let Ok(mut slot) = OUTPUT.write() {
        *slot = output.map(|write| Output { write, ansi });
    }
}

/// Writes each formatted event to the [`set_log_output`] callback or stderr.
#[cfg(not(target_arch = "wasm32"))]
struct OutputWriter;

#[cfg(not(target_arch = "wasm32"))]
impl std::io::Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(slot) = OUTPUT.read()
            && let Some(output) = slot.as_ref()
        {
            (output.write)(&String::from_utf8_lossy(buf));
            return Ok(buf.len());
        }
        std::io::stderr().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

static LISTENER: RwLock<Option<MessageListener>> = RwLock::new(None);

/// Calls `listener` with every warning and error that passes the log level,
/// besides the regular output; replaces an earlier listener, `None` removes
/// it.
pub fn set_message_listener(listener: Option<MessageListener>) {
    if let Ok(mut slot) = LISTENER.write() {
        *slot = listener;
    }
}

/// Forwards warnings and errors to the [`set_message_listener`] callback.
struct ListenerLayer;

impl<S: Subscriber> Layer<S> for ListenerLayer {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let level = match *event.metadata().level() {
            Level::ERROR => LogLevel::Error,
            Level::WARN => LogLevel::Warn,
            _ => return,
        };
        let Ok(slot) = LISTENER.read() else {
            return;
        };
        if let Some(listener) = slot.as_ref() {
            let mut text = MessageText::default();
            event.record(&mut text);
            listener(level, &text.0);
        }
    }
}

/// The message of an event, followed by its other fields as `name=value`.
#[derive(Default)]
struct MessageText(String);

impl Visit for MessageText {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.0.insert_str(0, &format!("{value:?}"));
        } else {
            self.0.push_str(&format!(" {}={value:?}", field.name()));
        }
    }
}

/// Initializes logging with the build default level (see the module
/// documentation); later calls do nothing.
pub fn init_logging() {
    init_logging_with_default(LogLevel::build_default());
}

/// Initializes logging with `default` as the level when `AVALANCHERS_LOG`
/// and `RUST_LOG` are unset; later calls do nothing.
pub fn init_logging_with_default(default: LogLevel) {
    let (filter, rejected) = filter_from_env(default);
    if install(filter) {
        if let Some((variable, value)) = rejected {
            tracing::warn!(
                "{variable}='{value}' is not a valid log level or filter; using the level {default}"
            );
        }
        tracing::debug!("Logging initialized");
    }
}

/// Sets the log level of the avalanchers crates (other crates stay at
/// `warn`), overriding `AVALANCHERS_LOG`; initializes logging first when
/// needed. Takes effect immediately, also for running simulations.
pub fn set_log_level(level: LogLevel) {
    let filter = EnvFilter::new(level.directives());
    match FILTER.get() {
        Some(handle) => {
            if let Err(error) = handle.reload(filter) {
                // only fails when the subscriber was dropped, i.e. never for
                // the global default installed here
                eprintln!("Could not change the log level to {level}: {error}");
            }
        }
        None => {
            install(filter);
        }
    }
}

/// Reads `AVALANCHERS_LOG` (then `RUST_LOG`): a level, or filter directives.
/// Returns the filter and the variable whose value was rejected.
fn filter_from_env(default: LogLevel) -> (EnvFilter, Option<(&'static str, String)>) {
    #[cfg(not(target_arch = "wasm32"))]
    for variable in [LOG_ENV_VAR, "RUST_LOG"] {
        let Ok(value) = std::env::var(variable) else {
            continue;
        };
        if value.trim().is_empty() {
            continue;
        }
        if let Ok(level) = value.parse::<LogLevel>() {
            return (EnvFilter::new(level.directives()), None);
        }
        return match EnvFilter::try_new(&value) {
            Ok(filter) => (filter, None),
            Err(_) => (
                EnvFilter::new(default.directives()),
                Some((variable, value)),
            ),
        };
    }
    (EnvFilter::new(default.directives()), None)
}

/// Installs the global subscriber with `filter`; returns false when a
/// subscriber was installed before (by this module or by the host program,
/// e.g. a test harness).
fn install(filter: EnvFilter) -> bool {
    if FILTER.get().is_some() {
        return false;
    }
    let (filter, handle) = reload::Layer::new(filter);
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(ListenerLayer);

    #[cfg(not(target_arch = "wasm32"))]
    let installed = {
        use std::io::IsTerminal;
        let redirected_ansi = OUTPUT
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|output| output.ansi));
        let color = redirected_ansi.unwrap_or_else(|| std::io::stderr().is_terminal())
            && std::env::var_os("NO_COLOR").is_none();
        registry
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(|| OutputWriter)
                    .with_ansi(color)
                    .with_target(true),
            )
            .try_init()
            .is_ok()
    };

    #[cfg(target_arch = "wasm32")]
    let installed = {
        // the filter above decides; the console layer passes everything on
        // and does not add performance marks for every message
        let config = tracing_wasm::WASMLayerConfigBuilder::new()
            .set_max_level(tracing::Level::TRACE)
            .set_report_logs_in_timings(false)
            .build();
        registry
            .with(tracing_wasm::WASMLayer::new(config))
            .try_init()
            .is_ok()
    };

    if installed {
        let _ = FILTER.set(handle);
    }
    installed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_case_insensitively_with_aliases() {
        for level in LogLevel::ALL {
            assert_eq!(level.as_str().parse::<LogLevel>(), Ok(level));
            assert_eq!(level.as_str().to_uppercase().parse::<LogLevel>(), Ok(level));
        }
        assert_eq!("Warning".parse::<LogLevel>(), Ok(LogLevel::Warn));
        assert_eq!(" none ".parse::<LogLevel>(), Ok(LogLevel::Off));
        assert!("verbose".parse::<LogLevel>().is_err());
    }

    #[test]
    fn verbose_levels_only_apply_to_own_crates() {
        assert_eq!(LogLevel::Warn.directives(), "warn");
        assert_eq!(LogLevel::Off.directives(), "off");
        let debug = LogLevel::Debug.directives();
        assert!(debug.starts_with("warn,"));
        assert!(debug.contains(",compute_core=debug"));
        assert!(!debug.contains("wgpu"));
        // every directive set is a valid filter
        for level in LogLevel::ALL {
            EnvFilter::try_new(level.directives()).unwrap();
        }
    }

    #[test]
    fn message_text_puts_the_message_first() {
        use tracing_subscriber::layer::SubscriberExt;

        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        impl<S: Subscriber> Layer<S> for Capture {
            fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
                let mut text = MessageText::default();
                event.record(&mut text);
                self.0.lock().unwrap().push(text.0);
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Capture(captured.clone()));
        tracing::subscriber::with_default(subscriber, || {
            let cells = 3;
            tracing::warn!(cells, "Release area is {} cells", 12);
        });
        assert_eq!(
            *captured.lock().unwrap(),
            ["Release area is 12 cells cells=3"]
        );
    }

    #[test]
    fn release_builds_default_to_warnings() {
        let expected = if cfg!(debug_assertions) {
            LogLevel::Info
        } else {
            LogLevel::Warn
        };
        assert_eq!(LogLevel::build_default(), expected);
    }
}
