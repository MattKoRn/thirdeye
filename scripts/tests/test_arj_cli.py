"""Synthetic ARJ fixtures for the native installer decoder (no game assets)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install


def arj_volume(name, payload, flags=0):
    """Make a minimal ARJ volume with a stored file and correct data CRC."""
    name = name.encode("ascii") + b"\0"
    header = bytearray(30)
    header[0] = 30                  # size of fixed fields
    header[4] = flags               # volume/continuation flags
    header[5] = 0                   # stored (no compression)
    header[6] = 0                   # normal file
    struct.pack_into("<I", header, 12, len(payload))
    struct.pack_into("<I", header, 16, len(payload))
    struct.pack_into("<I", header, 20, zlib.crc32(payload))
    header.extend(name)
    return (b"\x60\xea" + struct.pack("<H", len(header))
            + header + struct.pack("<I", zlib.crc32(header))
            + b"\0\0" + payload + b"\x60\xea\0\0")


class NativeArjTests(unittest.TestCase):
    def decoder(self):
        executable = install.unarj_executable()
        if not executable.is_file():
            self.skipTest("Build thirdeye-unarj first")
        return executable

    def test_multivolume_continuation_and_save_folders(self):
        exe = self.decoder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archives = []
            for name, member, payload, flags in [
                ("DATA1.ARJ", "EYE.RES", b"Hello ", 0x04),
                ("DATA2.ARJ", "EYE.RES", b"world", 0x08),
                ("DATA5.ARJ", "SLOT.BIN", b"save", 0),
                ("DATA6.ARJ", "HERO.DAT", b"character", 0),
            ]:
                source = root / name
                source.write_bytes(arj_volume(member, payload, flags))
                archives.append(source)
            out = root / "Output with spaces"
            subprocess.run([str(exe), str(out), "eob3", *map(str, archives)],
                           check=True, capture_output=True, text=True)
            self.assertEqual((out / "EYE.RES").read_bytes(), b"Hello world")
            self.assertEqual((out / "SAVEGAME/SLOT.BIN").read_bytes(), b"save")
            self.assertEqual((out / "CHARGEN/HERO.DAT").read_bytes(), b"character")

    def test_dungeon_hack_nested_savegame_file(self):
        """The retail DATA1.ARJ contains SAVEGAME/PC.DAT."""
        exe = self.decoder()
        for member in ("SAVEGAME/PC.DAT", r"SAVEGAME\PC.DAT"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "DATA1.ARJ"
                source.write_bytes(arj_volume(member, b"saved settings"))
                out = root / "output"
                subprocess.run([str(exe), str(out), "hack", str(source)],
                               check=True, capture_output=True, text=True)
                self.assertEqual((out / "SAVEGAME" / "PC.DAT").read_bytes(),
                                 b"saved settings")

    def test_reject_unsafe_nested_paths_before_writing(self):
        exe = self.decoder()
        names = (
            "../escape.txt", "SAVEGAME/../../escape.txt",
            "/absolute.txt", r"\absolute.txt", "C:/escape.txt",
            r"C:\escape.txt", "SAVEGAME//PC.DAT",
            "SAVEGAME/../PC.DAT", r"SAVEGAME\..\PC.DAT",
            "SAVEGAME/./PC.DAT", "SAVEGAME/CON/PC.DAT",
            "SAVEGAME/NUL.TXT", "SAVEGAME/PC.DAT/",
            "SAVEGAME/PC.DAT:stream", "SAVEGAME/TRAIL. /PC.DAT",
        )
        for member in names:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "DATA1.ARJ"
                source.write_bytes(arj_volume(member, b"unsafe"))
                out = root / "output"
                result = subprocess.run([str(exe), str(out), "hack", str(source)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Unsafe ARJ member name", result.stderr)
                self.assertFalse(out.exists(), member)

    def test_reject_normalized_duplicate_and_file_directory_conflict(self):
        exe = self.decoder()
        for first, second in (
            ("SAVEGAME/PC.DAT", r"savegame\pc.dat"),
            ("SAVEGAME", "SAVEGAME/PC.DAT"),
        ):
            with self.subTest(first=first, second=second), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = []
                for num, member in enumerate((first, second), 1):
                    path = root / f"DATA{num}.ARJ"
                    path.write_bytes(arj_volume(member, b"data"))
                    paths.append(path)
                out = root / "output"
                result = subprocess.run(
                    [str(exe), str(out), "hack", *map(str, paths)],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(out.exists())

    def test_reject_symlink_in_output_tree(self):
        exe = self.decoder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "DATA1.ARJ"
            source.write_bytes(arj_volume("SAVEGAME/PC.DAT", b"unsafe"))
            out = root / "output"
            out.mkdir()
            outside = root / "outside"
            outside.mkdir()
            try:
                (out / "SAVEGAME").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Creating directory symlinks is not permitted")
            result = subprocess.run([str(exe), str(out), "hack", str(source)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsafe ARJ output symlink", result.stderr)
            self.assertFalse((outside / "PC.DAT").exists())

    def test_reject_path_traversal_before_writing(self):
        exe = self.decoder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "DATA1.ARJ"
            source.write_bytes(arj_volume("../escape.txt", b"unsafe"))
            out = root / "output"
            result = subprocess.run([str(exe), str(out), "eob3", str(source)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "escape.txt").exists())
