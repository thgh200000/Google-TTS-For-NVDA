"""Download the Node.js runtime and place ``node.exe`` inside the add-on package.

One implementation shared by ``build.bat``, ``build.sh`` and the GitHub Actions
build workflow, so the three cannot drift.

The add-on drives Google's WASM TTS engine through a bundled Node.js executable
instead of a headless Chromium browser (see
``googleTtsForNvda/synthDrivers/googleTtsForNvda/node_bridge.py``), so the runtime
has to ship inside the ``.nvda-addon``. Node publishes per-platform archives; the
Windows x64 one is a full distribution whose executable sits at the top level of a
versioned folder::

    node-v22.23.3-win-x64/
    node-v22.23.3-win-x64/node.exe
    node-v22.23.3-win-x64/LICENSE
    ... npm, corepack, headers, docs ...

Only ``node.exe`` is packaged. Because that folder name tracks the version rather
than being a stable contract, the executable is located by a recursive search for
the expected file name instead of a hardcoded path.

The version is **pinned** and the downloaded archive is verified against the
``SHASUMS256.txt`` that Node publishes alongside it, so a build is reproducible and
a tampered or truncated download is rejected rather than packaged.

Why v22: it is an LTS release, and the last LTS line for which Node still publishes
a 32-bit Windows build (``win-x86``), which matters if 32-bit Windows support is
ever pursued. Node 24 and later dropped ``win-x86``.

Usage::

    python fetch_node_runtime.py
    python fetch_node_runtime.py --dest path/to/runtime --version v22.23.3
    python fetch_node_runtime.py --print-dest        # for build scripts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ADDON_ROOT = REPO_ROOT / "googleTtsForNvda"
DRIVER_DIR = ADDON_ROOT / "synthDrivers" / "googleTtsForNvda"

DEFAULT_VERSION = "v22.23.3"
DIST_BASE = "https://nodejs.org/dist"
# NVDA is Windows-only, and the add-on ships a single runtime binary.
DEFAULT_ARCH = "x64"
EXECUTABLE_NAME = "node.exe"
# Guard against a truncated download or an HTML error page being packaged.
MINIMUM_EXECUTABLE_BYTES = 20 * 1024 * 1024
PE_MAGIC = b"MZ"
VERSION_FILE_NAME = "node_version.json"
USER_AGENT = "GoogleTtsForNvda-Build/1.0"
CHUNK_BYTES = 1024 * 1024


def default_destination() -> Path:
    """Where the add-on expects the runtime, matching ``node_bridge.RUNTIME_DIR``."""
    return DRIVER_DIR / "runtime"


def archive_name(version: str = DEFAULT_VERSION, arch: str = DEFAULT_ARCH) -> str:
    return f"node-{version}-win-{arch}.zip"


def archive_url(version: str = DEFAULT_VERSION, arch: str = DEFAULT_ARCH) -> str:
    return f"{DIST_BASE}/{version}/{archive_name(version, arch)}"


def shasums_url(version: str = DEFAULT_VERSION) -> str:
    return f"{DIST_BASE}/{version}/SHASUMS256.txt"


def is_pe_executable(path: Path) -> bool:
    """Whether ``path`` looks like a Windows PE image rather than an error page."""
    try:
        with path.open("rb") as stream:
            return stream.read(2) == PE_MAGIC
    except OSError:
        return False


def find_node_executable(root: Path, executableName: str = EXECUTABLE_NAME) -> Path | None:
    """Recursively locate the runtime executable inside an extracted archive.

    Prefers the shallowest match, so the top-level ``node.exe`` wins over anything
    nested inside ``node_modules``.
    """
    if not root.is_dir():
        return None
    matches = [path for path in root.rglob("*") if path.is_file() and path.name == executableName]
    if not matches:
        return None
    return min(matches, key=lambda path: (len(path.parts), str(path)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, timeout: int = 600) -> Path:
    """Download ``url`` to ``destination``, streaming to a temporary file first."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporaryName = tempfile.mkstemp(dir=str(destination.parent), suffix=".download")
    os.close(handle)
    temporary = Path(temporaryName)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, CHUNK_BYTES)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def expected_archive_sha256(version: str, archiveName: str) -> str | None:
    """Read the checksum Node publishes for this archive, if it can be fetched."""
    try:
        request = urllib.request.Request(shasums_url(version), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for line in text.splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == archiveName:
            return digest.strip().lower()
    return None


def extract_archive(archive: Path, targetDir: Path) -> Path:
    """Unzip ``archive`` into ``targetDir``, rejecting entries that escape it."""
    targetDir.mkdir(parents=True, exist_ok=True)
    resolvedTarget = targetDir.resolve()
    with zipfile.ZipFile(archive) as zipFile:
        for member in zipFile.infolist():
            candidate = (resolvedTarget / member.filename).resolve()
            if resolvedTarget != candidate and resolvedTarget not in candidate.parents:
                raise RuntimeError(f"Refusing to extract a path-traversal member: {member.filename}")
        zipFile.extractall(resolvedTarget)
    return resolvedTarget


def probe_version(executable: Path) -> str:
    """Best-effort ``node --version``; only works when the host can run the binary."""
    try:
        import subprocess

        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().splitlines()[0].strip() if completed.stdout else ""


def install(
    *,
    version: str = DEFAULT_VERSION,
    arch: str = DEFAULT_ARCH,
    url: str | None = None,
    destination: Path | None = None,
    cacheDir: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download, verify, extract and place ``node.exe``. Returns the installed path."""

    def say(message: str) -> None:
        if not quiet:
            print(message)

    destDir = Path(destination) if destination is not None else default_destination()
    installed = destDir / EXECUTABLE_NAME
    if installed.is_file() and not force:
        if installed.stat().st_size >= MINIMUM_EXECUTABLE_BYTES and is_pe_executable(installed):
            say(f"      Already present: {installed} ({installed.stat().st_size:,} bytes)")
            return installed
        say(f"      Replacing invalid runtime at {installed}")

    downloadUrl = url or archive_url(version, arch)
    name = archive_name(version, arch) if url is None else downloadUrl.rstrip("/").rsplit("/", 1)[-1]
    cache = Path(cacheDir) if cacheDir is not None else Path(tempfile.gettempdir()) / "googleTtsForNvda-node-cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name

    if archive.is_file() and archive.stat().st_size > 0 and not force:
        say(f"      Using cached archive: {archive}")
    else:
        say(f"      Downloading {downloadUrl}")
        download(downloadUrl, archive)
        say(f"      Downloaded {archive.stat().st_size:,} bytes")

    # Verify against the checksum Node publishes next to the release.
    if url is None:
        expected = expected_archive_sha256(version, name)
        actual = sha256(archive)
        if expected is None:
            say("      WARNING: could not fetch SHASUMS256.txt; skipped archive verification")
        elif expected != actual:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {expected}, got {actual}. "
                "The cached archive was deleted; re-run the build."
            )
        else:
            say(f"      Verified against SHASUMS256.txt: {actual}")

    with tempfile.TemporaryDirectory(prefix="gtts-node-") as stagingName:
        staging = Path(stagingName)
        extract_archive(archive, staging)
        found = find_node_executable(staging)
        if found is None:
            names = sorted({path.name for path in staging.rglob("*") if path.is_file()})[:20]
            raise RuntimeError(f"{EXECUTABLE_NAME} was not found inside {archive.name}. Archive contained: {names}")
        size = found.stat().st_size
        if size < MINIMUM_EXECUTABLE_BYTES:
            raise RuntimeError(f"{found.name} is only {size:,} bytes; expected at least {MINIMUM_EXECUTABLE_BYTES:,}.")
        if not is_pe_executable(found):
            raise RuntimeError(f"{found.name} is not a Windows PE executable.")
        destDir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, installed)
        with suppress(OSError):
            os.chmod(installed, 0o755)
        say(f"      Installed {installed} ({installed.stat().st_size:,} bytes)")

    metadata = {
        "version": version,
        "arch": arch,
        "url": downloadUrl,
        "archiveSha256": sha256(archive),
        "executable": EXECUTABLE_NAME,
        "executableSha256": sha256(installed),
        "executableBytes": installed.stat().st_size,
        "reportedVersion": probe_version(installed),
        "license": "nodeRuntime/LICENSE-node.txt",
    }
    (destDir / VERSION_FILE_NAME).write_text(json.dumps(metadata, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    say(f"      Node version: {metadata['reportedVersion'] or 'unknown (host cannot execute a Windows binary)'}")
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Node.js version to bundle (default: %(default)s).")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=("x64", "x86", "arm64"), help="Windows architecture.")
    parser.add_argument("--url", default=None, help="Override the archive URL entirely (skips checksum verification).")
    parser.add_argument(
        "--dest",
        default=None,
        help=f"Destination directory (default: {default_destination().relative_to(REPO_ROOT)}).",
    )
    parser.add_argument("--cache-dir", default=None, help="Directory to cache the downloaded archive in.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if already installed.")
    parser.add_argument("--quiet", action="store_true", help="Only report errors.")
    parser.add_argument("--print-dest", action="store_true", help="Print the destination directory and exit.")
    args = parser.parse_args(argv)

    if args.print_dest:
        print(default_destination() if args.dest is None else Path(args.dest))
        return 0

    try:
        install(
            version=args.version,
            arch=args.arch,
            url=args.url,
            destination=Path(args.dest) if args.dest else None,
            cacheDir=Path(args.cache_dir) if args.cache_dir else None,
            force=args.force,
            quiet=args.quiet,
        )
    except (urllib.error.URLError, RuntimeError, OSError, zipfile.BadZipFile) as error:
        print(f"[ERROR] Could not install the Node.js runtime: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
