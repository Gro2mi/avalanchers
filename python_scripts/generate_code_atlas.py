#!/usr/bin/env python3
# @atlas: Scans the repo for per-file summary lines (Rust `//!` module docs, `@atlas:` comments elsewhere) and regenerates code_atlas.md — the auto-generated successor to the hand-written file atlas, modeled on Hexagons/bundle.py.
"""Regenerate code_atlas.md from `@atlas:` comments scattered through the source tree.

Each source file may carry a one-line comment near its top:

    //! <description>            (.rs — a normal Rust inner doc comment, so
                                  `cargo doc` renders it too; must be line 1)
    // @atlas: <description>      (.wgsl, .js)
    # @atlas: <description>       (.py, shell; after the shebang if present)

This script finds the first such comment in every scanned file and rebuilds
code_atlas.md's tables from it. It does not read or write anything else about
a file's content.

Usage
─────
    python3 python_scripts/generate_code_atlas.py

Re-running is idempotent: given the same `@atlas:` comments, the generated
file is byte-for-byte identical.
"""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS_PATH = REPO_ROOT / "code_atlas.md"

SCAN_EXTS = {".rs", ".wgsl", ".py", ".js"}
SKIP_DIRS = {"target", "node_modules", ".git"}

ATLAS_RE = re.compile(r"^\s*(?://|#)\s*@atlas:\s*(.+?)\s*$", re.MULTILINE)

# Rust carries its summary as a normal inner doc comment (`//!`) on the first
# line instead of a bespoke marker, so `cargo doc` renders the same sentence
# that the atlas tabulates. Only the first line is the atlas entry; any further
# `//!` lines are the module's own prose and are left alone.
RUST_DOC_RE = re.compile(r"\A//!\s*(.+?)\s*$", re.MULTILINE)


def summary_for(rel_path: str, content: str):
    """The one-line description for a file, or None if it carries none."""
    if rel_path.endswith(".rs"):
        m = RUST_DOC_RE.match(content)
        if m:
            return m.group(1).strip()
        # tolerate the pre-`//!` marker so an un-migrated file still lands
        m = ATLAS_RE.search(content)
        return m.group(1).strip() if m else None
    m = ATLAS_RE.search(content)
    return m.group(1).strip() if m else None

PREAMBLE = """\
# Code atlas

One line per source file, so a newcomer — or an agent — can find the right file
without grepping. Code only: what each file is, not what the project has found.

This file is generated. To change an entry, edit the source file's summary line
— in Rust the first `//!` doc comment, elsewhere an `@atlas: ...` comment — and
run `python3 python_scripts/generate_code_atlas.py`; the pre-commit hook
(`.githooks/pre-commit`) does this automatically on every commit.
"""


class Section:
    """One markdown table in the atlas.

    `prefixes` is a list of repo-relative path prefixes (directories, ending
    in "/") or exact file paths that belong to this section. `exclude` is an
    optional list of prefixes that would otherwise match but belong to a more
    specific section instead (e.g. the shaders subdir of compute_core).
    `root` is the prefix stripped off each path for display; "" shows the
    full repo-relative path.
    """

    def __init__(self, title, prefixes, root, col_header="what it is", intro=None, exclude=None):
        self.title = title
        self.prefixes = prefixes
        self.exclude = exclude or []
        self.root = root
        self.col_header = col_header
        self.intro = intro
        self.files = []  # (display_path, description)

    def matches(self, rel_path: str) -> bool:
        for ex in self.exclude:
            if rel_path.startswith(ex):
                return False
        for p in self.prefixes:
            if p.endswith("/"):
                if rel_path.startswith(p):
                    return True
            elif rel_path == p:
                return True
        return False

    def display_path(self, rel_path: str) -> str:
        if self.root and rel_path.startswith(self.root):
            return rel_path[len(self.root):]
        return rel_path


SECTIONS = [
    Section(
        title="crates/compute_core — GPU orchestration and physics",
        prefixes=["crates/compute_core/"],
        exclude=["crates/compute_core/src/shaders/"],
        root="crates/compute_core/",
    ),
    Section(
        title="crates/compute_core/src/shaders — WGSL",
        prefixes=["crates/compute_core/src/shaders/"],
        root="crates/compute_core/src/shaders/",
        col_header="what it does",
        intro=(
            "`utils.wgsl` is textually inlined into every other shader, so its `SimSettings`\n"
            "struct must stay byte-compatible with `settings.rs`."
        ),
    ),
    Section(
        title="crates/simulation",
        prefixes=["crates/simulation/"],
        root="crates/simulation/",
    ),
    Section(
        title="crates/data_processor",
        prefixes=["crates/data_processor/"],
        root="crates/data_processor/",
    ),
    Section(
        title="crates/cli",
        prefixes=["crates/cli/"],
        root="crates/cli/",
    ),
    Section(
        title="Bindings and frontend",
        prefixes=[
            "crates/python_bindings/",
            "crates/wasm_bindings/",
            "python_module/",
            "avalanchers_example.py",
            "frontend/",
        ],
        root="",
    ),
    Section(
        title="python_scripts",
        prefixes=["python_scripts/"],
        root="python_scripts/",
    ),
    Section(
        title="campaign/analysis — one-shot analysis scripts",
        prefixes=["campaign/analysis/"],
        root="campaign/analysis/",
        col_header="what it measures",
        intro=(
            "These read scratchpad inputs (raster dumps, per-event calibration output) that\n"
            "are **not** in the repo; they are kept so the method survives, not to re-run\n"
            "unmodified. Set `D` at the top of a script to a directory holding the inputs."
        ),
    ),
]

OTHER = Section(title="Other", prefixes=[], root="")


def is_skipped_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def scan_files():
    for dirpath, dirnames, filenames in __import__("os").walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if not is_skipped_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix in SCAN_EXTS:
                yield fpath


def git_tracked_files():
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True, text=True, check=True,
        )
        return set(out.stdout.splitlines())
    except Exception:
        return None


def main():
    tagged = {}
    untagged = []

    for fpath in scan_files():
        rel = str(fpath.relative_to(REPO_ROOT))
        try:
            content = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            untagged.append(rel)
            continue
        desc = summary_for(rel, content)
        if desc:
            tagged[rel] = desc
        else:
            untagged.append(rel)

    for rel, desc in tagged.items():
        placed = False
        for section in SECTIONS:
            if section.matches(rel):
                section.files.append((section.display_path(rel), desc))
                placed = True
                break
        if not placed:
            OTHER.files.append((rel, desc))

    all_sections = SECTIONS + ([OTHER] if OTHER.files else [])

    lines = [PREAMBLE]
    for section in all_sections:
        lines.append(f"## {section.title}\n")
        if section.intro:
            lines.append(section.intro + "\n")
        lines.append(f"| file | {section.col_header} |")
        lines.append("|---|---|")
        for path, desc in sorted(section.files, key=lambda x: x[0]):
            lines.append(f"| `{path}` | {desc} |")
        lines.append("")

    ATLAS_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")

    total = sum(len(s.files) for s in all_sections)
    print(f"Wrote {total} entries across {len(all_sections)} sections to {ATLAS_PATH.relative_to(REPO_ROOT)}")

    if untagged:
        tracked = git_tracked_files()
        if tracked is not None:
            untagged_tracked = sorted(r for r in untagged if r in tracked)
            untagged_other = sorted(r for r in untagged if r not in tracked)
        else:
            untagged_tracked, untagged_other = sorted(untagged), []

        if untagged_tracked:
            print(f"\nWARNING: {len(untagged_tracked)} tracked source file(s) have no @atlas tag:")
            for rel in untagged_tracked:
                print(f"  {rel}")
        if untagged_other:
            print(f"\n{len(untagged_other)} untracked/generated file(s) also have no @atlas tag (informational):")
            for rel in untagged_other:
                print(f"  {rel}")


if __name__ == "__main__":
    main()
