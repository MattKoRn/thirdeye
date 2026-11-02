#!/usr/bin/env python3
"""One-command, per-user installer for the Thirdeye engine and owned game data.

No copyrighted data is distributed with Thirdeye. Archive.org downloads are
opt-in and require the operator to confirm their right to download the files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
GAMES = {
    "eob3": ("Eye of the Beholder III", "EYE.RES", "eye-of-the-beholder-3"),
    "hack": ("Dungeon Hack", "HACK.RES", "DungeonHackUSA"),
}
MAX_DOWNLOAD = 2 * 1024**3
MAX_UNPACKED = 3 * 1024**3


def info(message):
    print("[thirdeye] " + str(message), flush=True)


def fail(message):
    raise RuntimeError(str(message))


def ask(message, options):
    while True:
        print(message)
        for key, label in options.items():
            print("  {}. {}".format(key, label))
        answer = input("Choice: ").strip().lower()
        if answer in options:
            return answer
        info("Please enter one of: " + ", ".join(options))


def confirm_rights():
    info("Internet Archive uploads may be unauthorized. Archive.org hosting")
    info("and abandonware labels DO NOT grant copyright permission.")
    return input("Confirm you have the legal right to download these assets (type I AGREE): ").strip() == "I AGREE"


def run(args, **kwargs):
    info("Running: " + " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def os_name():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "mac"
    if os.name == "nt":
        return "windows"
    fail("Unsupported platform: " + sys.platform)


def app_root(kind):
    home = Path.home()
    if kind == "windows":
        return Path(os.environ.get("LOCALAPPDATA", str(home / "AppData/Local"))) / "Thirdeye"
    if kind == "mac":
        return home / "Library/Application Support/Thirdeye"
    return Path(os.environ.get("XDG_DATA_HOME", str(home / ".local/share"))) / "thirdeye"


def ensure_dependencies(kind):
    """Provision platform prerequisites with the native package manager."""
    if kind == "linux":
        if not shutil.which("apt-get"):
            fail("Automatic dependencies currently require Debian/Ubuntu with apt. "
                 "Install CMake, Ninja, Git, a C++20 compiler, SDL3 and OpenAL manually, "
                 "then use --no-deps.")
        elevate = [] if os.geteuid() == 0 else ["sudo"]
        if elevate and not shutil.which("sudo"):
            fail("Install sudo or run as root to install apt dependencies.")
        run(elevate + ["apt-get", "update"])
        run(elevate + ["apt-get", "install", "-y",
                       "build-essential", "cmake", "ninja-build", "git",
                       "libsdl3-dev", "libopenal-dev", "7zip", "python3"])
    elif kind == "mac":
        if not shutil.which("brew"):
            fail("Install Homebrew from https://brew.sh first, then rerun. "
                 "Xcode Command Line Tools are also required (xcode-select --install).")
        run(["brew", "install", "cmake", "ninja", "git", "sdl3", "openal-soft", "sevenzip"])
    else:
        if not shutil.which("winget"):
            fail("Windows App Installer (winget) is needed for automatic setup.")
        for cmd, package in (("git", "Git.Git"), ("cmake", "Kitware.CMake"),
                             ("ninja", "Ninja-build.Ninja"), ("7z", "7zip.7zip")):
            if not shutil.which(cmd):
                run(["winget", "install", "--id", package, "-e",
                     "--accept-package-agreements", "--accept-source-agreements"])
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        has_vs = vswhere.is_file() and subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True).stdout.strip()
        if not has_vs:
            info("Installing Visual Studio C++ Build Tools; Windows may request elevation.")
            run(["winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools",
                 "-e", "--override",
                 "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
                 "--accept-package-agreements", "--accept-source-agreements"])


def vcpkg_toolchain(home):
    root = Path(os.environ.get("VCPKG_ROOT", str(home / "vcpkg")))
    exe = root / "vcpkg.exe"
    if not exe.exists():
        if not shutil.which("git"):
            fail("Git is missing; reopen your terminal after winget has installed it.")
        if not root.exists():
            run(["git", "clone", "https://github.com/microsoft/vcpkg.git", str(root)])
        run(["cmd", "/c", str(root / "bootstrap-vcpkg.bat")])
    run([str(exe), "install", "sdl3", "openal-soft", "--triplet", "x64-windows"])
    return root / "scripts/buildsystems/vcpkg.cmake", root / "installed/x64-windows/bin"


def build_engine(kind, home):
    build = ROOT / "build" / "installer"
    cmake = shutil.which("cmake")
    if not cmake:
        fail("CMake not on PATH. Reopen your terminal after dependency installation.")
    options = ["-DUNIT_TESTS=OFF", "-DBUILD_LAUNCHER=OFF",
               "-DBUILD_ARC=OFF", "-DBUILD_DAESOP=OFF"]
    vcpkg_bin = None
    if kind == "windows":
        toolchain, vcpkg_bin = vcpkg_toolchain(home)
        args = [cmake, "-S", ROOT, "-B", build, "-G", "Visual Studio 17 2022",
                "-A", "x64", "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain)] + options
    else:
        args = [cmake, "-S", ROOT, "-B", build, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release"] + options
    run(args)
    run([cmake, "--build", build, "--target", "thirdeye", "--config", "Release",
         "--parallel", "2"])
    candidates = ([build / "thirdeye.app/Contents/MacOS/thirdeye"] if kind == "mac" else
                  [build / "Release/thirdeye.exe", build / "thirdeye.exe"] if kind == "windows" else
                  [build / "thirdeye"])
    executable = next((p for p in candidates if p.is_file()), None)
    if executable is None:
        fail("Build completed but no engine binary found: " + repr(candidates))
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    installed = bin_dir / executable.name
    shutil.copy2(executable, installed)
    if kind == "windows":
        for dll_dir in [vcpkg_bin, build, build / "Release"]:
            if dll_dir and dll_dir.is_dir():
                for dll in dll_dir.glob("*.dll"):
                    shutil.copy2(dll, bin_dir / dll.name)
        for dll in build.rglob("*wildmidi*.dll"):
            shutil.copy2(dll, bin_dir / dll.name)
    else:
        installed.chmod(installed.stat().st_mode | stat.S_IXUSR)
    run([installed, "--version"])
    return installed


def valid_game(directory, game):
    if not Path(directory).is_dir():
        return False
    names = {p.name.upper() for p in Path(directory).iterdir() if p.is_file()}
    return GAMES[game][1] in names


def find_game_dir(source, game, depth=5):
    source = Path(source)
    if not source.is_dir():
        return None
    if valid_game(source, game):
        return source.resolve()
    base_depth = len(source.parts)
    for parent, dirs, _ in os.walk(source, followlinks=False):
        here = Path(parent)
        dirs[:] = [d for d in dirs if not d.startswith(".") and
                   len(here.parts) - base_depth < depth]
        if valid_game(here, game):
            return here.resolve()
    return None


def discover_installs(home, game):
    user = Path.home()
    paths = [home / "games" / game, ROOT / "data", ROOT.parent / "data",
             user / "Games", user / "GOG Games", user / "Downloads"]
    if os.name == "nt":
        for env in ("ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(env):
                paths.append(Path(os.environ[env]) / "GOG Galaxy/Games")
    for path in paths:
        result = find_game_dir(path, game, depth=3)
        if result:
            return result
    return None


def safe_zip_extract(source, destination):
    """Extract without traversal, symlinks, or unbounded expansion."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        total = 0
        entries = archive.infolist()
        if len(entries) > 30000:
            fail("Archive has too many entries")
        for member in entries:
            total += member.file_size
            if total > MAX_UNPACKED:
                fail("Archive is too large when unpacked")
            name = member.filename.replace("\\", "/")
            parts = Path(name).parts
            if (not parts or name.startswith("/") or
                re.match(r"^[A-Za-z]:", name) or ".." in parts or
                (member.external_attr >> 16) & 0o170000 == 0o120000):
                fail("Unsafe archive member: " + name)
            target = destination.joinpath(*parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def seven_zip():
    for name in ("7z", "7zz", "7za"):
        exe = shutil.which(name)
        if exe:
            return exe
    if os.name == "nt":
        for env in ("ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(env):
                candidate = Path(os.environ[env]) / "7-Zip/7z.exe"
                if candidate.is_file():
                    return str(candidate)
    return None


def extract_external(source, destination):
    extractor = seven_zip()
    if not extractor:
        fail("7-Zip is needed to unpack ISO/ARJ files. Install 7zip, then rerun.")
    Path(destination).mkdir(parents=True, exist_ok=True)
    run([extractor, "x", "-y", "-o" + str(destination), str(source)])


def raw_bin_to_iso(source, target):
    """Convert MODE1/2352 CD sectors to a normal 2048-byte ISO image."""
    with Path(source).open("rb") as src, Path(target).open("wb") as out:
        while True:
            sector = src.read(2352)
            if not sector:
                break
            if len(sector) != 2352 or sector[:1] != b"\x00" or sector[15] != 1:
                fail("BIN image is not MODE1/2352: " + str(source))
            out.write(sector[16:2064])


def expand_archives(root, game):
    """Expand installer ZIPs, optional CD image, then original ARJ data sets."""
    root = Path(root)
    # Keep all unpacked data in a temporary workspace, never overwrite saves.
    # Only BIN files paired with a CUE sheet are CD images. Game saves also
    # use .BIN, and must NEVER be interpreted as 2352-byte raw CD sectors.
    if game == "hack":
        for cue in root.rglob("*"):
            if not cue.is_file() or cue.suffix.lower() != ".cue":
                continue
            image = next((p for p in cue.parent.iterdir()
                          if p.is_file() and p.suffix.lower() == ".bin"
                          and p.stem.casefold() == cue.stem.casefold()), None)
            if image is None:
                continue
            if image.stat().st_size > MAX_DOWNLOAD:
                fail("CD image exceeds permitted size")
            iso = image.with_suffix(".converted.iso")
            raw_bin_to_iso(image, iso)
            extract_external(iso, root / "cd_contents")
    for image in list(root.rglob("*")):
        if image.is_file() and image.suffix.lower() == ".iso" and not image.name.endswith(".converted.iso"):
            extract_external(image, root / "cd_contents")
    arjs = sorted((p for p in root.rglob("*") if p.is_file() and
                   re.fullmatch(r"DATA[0-9]+\.ARJ", p.name, flags=re.I)),
                  key=lambda p: int(re.search(r"[0-9]+", p.name).group()))
    if arjs:
        out = root / "unpacked_game"
        for arj in arjs:
            number = int(re.search(r"[0-9]+", arj.name).group())
            subfolder = ("SAVEGAME" if number == 5 else
                         "CHARGEN" if number == 6 else "") if game == "eob3" else ""
            extract_external(arj, out / subfolder)
    return find_game_dir(root / "unpacked_game", game) or find_game_dir(root, game)


def archive_metadata(item):
    url = "https://archive.org/metadata/" + urllib.parse.quote(item, safe="")
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Thirdeye installer/1.0"}),
            timeout=45) as resp:
        return json.load(resp)


def archive_choices(game, metadata):
    files = {f.get("name", "").casefold(): f for f in metadata.get("files", [])
             if isinstance(f, dict)}
    if game == "eob3":
        expected = ["EOB3_Disk{}.zip".format(i) for i in range(1, 5)]
    else:
        expected = ["Dungeon Hack (USA).zip"]
    selected = [files.get(name.casefold()) for name in expected]
    if not all(selected):
        fail("The archive.org item no longer lists its expected game files. "
             "Use a local copy; no unverified file will be downloaded.")
    return selected


def archive_download(game, stage):
    item = GAMES[game][2]
    info("Inspecting https://archive.org/details/" + item)
    metadata = archive_metadata(item)
    selected = archive_choices(game, metadata)
    info("Archive description: " + str(metadata.get("metadata", {}).get("title", item)))
    info("Rights metadata (not proof of permission): " +
         str(metadata.get("metadata", {}).get("licenseurl", "none")))
    if not confirm_rights():
        fail("Archive download cancelled. You can import your own copy instead.")
    for f in selected:
        filename = f["name"]
        size = int(f.get("size", "0") or 0)
        if size > MAX_DOWNLOAD:
            fail("Archive file is too large: " + filename)
        url = "https://archive.org/download/" + urllib.parse.quote(item) + "/" + \
              urllib.parse.quote(filename)
        target = stage / Path(filename).name
        info("Downloading " + filename)
        digest = hashlib.sha1()
        count = 0
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Thirdeye installer/1.0"}),
                timeout=120) as response, target.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_DOWNLOAD:
                    fail("Download is larger than permitted")
                digest.update(chunk)
                out.write(chunk)
        if size and size != count:
            fail("Incomplete archive download: " + filename)
        if f.get("sha1") and digest.hexdigest().lower() != f["sha1"].lower():
            fail("Archive SHA-1 did not match: " + filename)
        safe_zip_extract(target, stage)
    return expand_archives(stage, game)


def source_directory(path, game, stage):
    src = Path(path).expanduser().resolve()
    if src.is_dir():
        found = find_game_dir(src, game)
        if not found:
            fail("No " + GAMES[game][1] + " found inside " + str(src))
        return found
    if not src.is_file():
        fail("Source does not exist: " + str(src))
    if src.suffix.lower() in (".zip", ".sh"):
        safe_zip_extract(src, stage)
    elif src.suffix.lower() in (".iso", ".arj"):
        extract_external(src, stage)
    elif src.suffix.lower() == ".bin":
        iso = stage / "disc.iso"
        raw_bin_to_iso(src, iso)
        extract_external(iso, stage)
    else:
        fail("Unsupported source. Provide a game directory, ZIP, GOG .sh, ISO, ARJ or BIN.")
    result = expand_archives(stage, game)
    if not result:
        fail("Could not locate " + GAMES[game][1] + " after extraction. "
             "The archive may require a different disk layout; import an installed game folder.")
    return result


def copy_game(source, destination, game):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        fail("Refusing to overwrite existing files or saves at " + str(destination))
    # Never follow archive symlinks (including symlinked directories).
    for parent, directories, files in os.walk(source, followlinks=False):
        for name in directories + files:
            if (Path(parent) / name).is_symlink():
                fail("Refusing to copy a symlink from a game archive: " + name)
    shutil.copytree(source, destination)
    # DOS assets can be lowercase on case-sensitive filesystems; normalize .RES.
    required = GAMES[game][1]
    matches = [x for x in destination.iterdir() if x.name.upper() == required]
    if matches and matches[0].name != required:
        matches[0].rename(destination / required)
    if not valid_game(destination, game):
        fail("Installation did not produce " + required)
    return destination


def make_shortcut(kind, home, executable, game, data):
    home = Path(home)
    icon = ROOT / "apps/launcher/resources/images/thirdeye.png"
    if icon.is_file():
        shutil.copy2(icon, home / "thirdeye.png")
    label = "Thirdeye - " + GAMES[game][0]
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    if kind == "windows":
        shortcut = desktop / (label + ".lnk")
        # Invoke PowerShell only for the Windows Shell Link COM API.
        script = (
            "$w=New-Object -ComObject WScript.Shell; "
            "$s=$w.CreateShortcut(" + ps_quote(str(shortcut)) + "); "
            "$s.TargetPath=" + ps_quote(str(executable)) + "; "
            "$s.Arguments=" + ps_quote('"' + str(data) + '" --game=' + game + ' --scale=3') + "; "
            "$s.WorkingDirectory=" + ps_quote(str(data)) + "; "
            "$s.Save()")
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
    else:
        import shlex
        cmd = "exec {} {} --game={} --scale=3 \"$@\"\n".format(
            shlex.quote(str(executable)), shlex.quote(str(data)), game)
        runner = home / ("launch-" + game + ".sh")
        runner.write_text("#!/bin/sh\n" + cmd, encoding="utf-8")
        runner.chmod(0o755)
        if kind == "mac":
            shortcut = desktop / (label + ".command")
            shutil.copy2(runner, shortcut)
            shortcut.chmod(0o755)
        else:
            shortcut = desktop / (label + ".desktop")
            quoted = str(runner).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
            shortcut.write_text(
                "[Desktop Entry]\nVersion=1.0\nType=Application\nName=" + label +
                "\nExec=/bin/sh \"" + quoted + "\"\n" +
                "Icon=" + str(home / "thirdeye.png") +
                "\nTerminal=false\nCategories=Game;RolePlaying;\n",
                encoding="utf-8")
            shortcut.chmod(0o755)
    return shortcut


def ps_quote(value):
    return "'" + value.replace("'", "''") + "'"


def select_game(home, requested):
    found = {game: discover_installs(home, game) for game in GAMES}
    for game, path in found.items():
        info("{} existing install: {}".format(GAMES[game][0], path or "not found"))
    if requested == "skip":
        choices = {key: GAMES[key][0] + " (reuse existing)" for key, v in found.items() if v}
        if not choices:
            fail("Skip requested, but no existing EYE.RES or HACK.RES installation was detected.")
        game = next(iter(choices)) if len(choices) == 1 else ask("Reuse which game?", choices)
        return game, found[game]
    if requested in GAMES:
        return requested, found[requested]
    choices = {key: value[0] for key, value in GAMES.items()}
    if any(found.values()):
        choices["s"] = "Skip choosing a new game; reuse an existing installation"
    answer = ask("Which game should this installation launch?", choices)
    if answer == "s":
        return select_game(home, "skip")
    return answer, found[answer]


def install_data(home, game, already, args):
    if already and not args.source and not args.archive:
        info("Existing install detected: " + str(already))
        if ask("Keep its assets and saved games?", {"y": "Yes — skip asset install",
                                                    "n": "Import another copy"}) == "y":
            return already
    if args.source:
        method = "local"
    elif args.archive:
        method = "archive"
    else:
        method = ask("How should game assets be supplied?",
                     {"l": "Import legally owned local folder/archive",
                      "a": "Check Internet Archive (explicit rights confirmation required)"} )
        method = "archive" if method == "a" else "local"
    with tempfile.TemporaryDirectory(prefix="thirdeye-assets-") as folder:
        stage = Path(folder)
        if method == "archive":
            src = archive_download(game, stage)
        else:
            path = args.source or input("Path to your installed game folder or archive: ").strip().strip('"')
            src = source_directory(path, game, stage)
        destination = home / "games" / game
        if destination.exists():
            if valid_game(destination, game):
                info("Already installed at " + str(destination) + "; not overwriting saves.")
                return destination
            fail("A non-game folder already exists at " + str(destination) +
                 ". Move it manually to avoid data loss.")
        return copy_game(src, destination, game)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and install Thirdeye + owned game assets")
    ap.add_argument("--game", choices=["eob3", "hack", "skip"],
                    help="Pick game; skip reuses an existing install")
    ap.add_argument("--source", help="Existing game folder, ZIP, GOG .sh, ISO, ARJ, or BIN")
    ap.add_argument("--archive", action="store_true",
                    help="Use Internet Archive metadata and opt-in download")
    ap.add_argument("--no-deps", action="store_true",
                    help="Build with previously installed dependencies")
    args = ap.parse_args(argv)
    if args.archive and args.source:
        ap.error("--archive and --source cannot be used together")
    kind = os_name()
    home = app_root(kind)
    home.mkdir(parents=True, exist_ok=True)
    info("Platform: {} ({})".format(kind, platform.machine()))
    game, already = select_game(home, args.game)
    if not args.no_deps:
        ensure_dependencies(kind)
    executable = build_engine(kind, home)
    data = install_data(home, game, already, args)
    shortcut = make_shortcut(kind, home, executable, game, data)
    config = {"game": game, "game_data": str(data), "engine": str(executable),
              "shortcut": str(shortcut)}
    (home / "installation.json").write_text(json.dumps(config, indent=2) + "\n")
    info("Verified " + GAMES[game][1] + " at " + str(data))
    info("Engine: " + str(executable))
    info("Desktop shortcut: " + str(shortcut))
    info("Installation complete. Launch via the desktop shortcut.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError,
            zipfile.BadZipFile, urllib.error.URLError) as exc:
        info("ERROR: " + str(exc))
        info("Fix the reported issue and rerun; existing game saves were not overwritten.")
        sys.exit(1)
