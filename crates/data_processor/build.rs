//! Records the commit of the build, which identifies the code that wrote a run.
//!
//! Sets for the crate:
//! - `AVALANCHERS_GIT_COMMIT`: full commit hash, or `unknown` without git;
//! - `AVALANCHERS_GIT_DIRTY`: `true` when tracked sources of the build differ from
//!   the commit (`crates/`, `Cargo.toml`, `Cargo.lock`);
//! - `AVALANCHERS_GIT_HASH`: the short hash, with `.dirty` appended when dirty, for
//!   the version string `0.2.0+<hash>`.
//!
//! A build without a git checkout (e.g. from a source archive or in CI) can pass
//! the commit in the environment variables `AVALANCHERS_GIT_COMMIT` and
//! `AVALANCHERS_GIT_DIRTY` instead.
//!
//! Release builds rerun this script whenever a tracked source changes, so the dirty
//! flag is exact. Debug builds only rerun it on commits, checkouts and staging, to
//! keep edit-compile cycles from rebuilding everything that depends on this crate.

use std::env;
use std::path::Path;
use std::process::Command;

/// Paths whose changes alter the build.
const SOURCES: [&str; 3] = ["crates", "Cargo.toml", "Cargo.lock"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AVALANCHERS_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=AVALANCHERS_GIT_DIRTY");

    let (commit, dirty) = match env::var("AVALANCHERS_GIT_COMMIT") {
        Ok(commit) if !commit.trim().is_empty() => {
            let dirty = env::var("AVALANCHERS_GIT_DIRTY")
                .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "yes"));
            (commit.trim().to_string(), dirty)
        }
        _ => from_git(),
    };
    let short: String = commit.chars().take(12).collect();
    let suffix = if dirty { ".dirty" } else { "" };
    println!("cargo:rustc-env=AVALANCHERS_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=AVALANCHERS_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=AVALANCHERS_GIT_HASH={short}{suffix}");
}

/// The commit of the workspace checkout and whether its build sources differ.
fn from_git() -> (String, bool) {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&workspace)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
    };

    let Some(commit) = git(&["rev-parse", "HEAD"]) else {
        return ("unknown".to_string(), false);
    };
    // A worktree or submodule has its git directory elsewhere.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        for name in ["HEAD", "logs/HEAD", "index"] {
            println!("cargo:rerun-if-changed={git_dir}/{name}");
        }
    }
    if env::var("PROFILE").is_ok_and(|profile| profile == "release") {
        let files = git(&[&["ls-files", "--"][..], &SOURCES[..]].concat()).unwrap_or_default();
        for file in files.lines() {
            println!("cargo:rerun-if-changed={}", workspace.join(file).display());
        }
    }
    let status = git(&[
        &["status", "--porcelain", "--untracked-files=no", "--"][..],
        &SOURCES[..],
    ]
    .concat());
    let dirty = status.is_some_and(|status| !status.is_empty());
    (commit, dirty)
}
