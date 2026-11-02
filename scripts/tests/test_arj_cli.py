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
