"""Dependency-free installer regression tests; no game data or downloads needed."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install


class InstallerTests(unittest.TestCase):
    def test_detect_games_and_skip_false_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eob = root / "nested" / "eye"
            eob.mkdir(parents=True)
            (eob / "EYE.RES").write_bytes(b"sample")
            self.assertEqual(install.find_game_dir(root, "eob3"), eob)
            self.assertIsNone(install.find_game_dir(root, "hack"))
            (eob / "HACK.RES").write_bytes(b"sample")
            self.assertEqual(install.find_game_dir(root, "hack"), eob)

    def test_extract_zip_and_reject_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.zip"
            with zipfile.ZipFile(good, "w") as archive:
                archive.writestr("data/EYE.RES", b"ok")
            install.safe_zip_extract(good, root / "out")
            self.assertTrue((root / "out/data/EYE.RES").exists())
            evil = root / "bad.zip"
            with zipfile.ZipFile(evil, "w") as archive:
                archive.writestr("../escape.txt", b"bad")
            with self.assertRaises(RuntimeError):
                install.safe_zip_extract(evil, root / "out")
            self.assertFalse((root / "escape.txt").exists())

    def test_bin_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "disc.bin"
            data = b"x" * 2048
            source.write_bytes(b"\0" * 15 + b"\1" + data + b"\0" * 288)
            target = root / "disc.iso"
            install.raw_bin_to_iso(source, target)
            self.assertEqual(target.read_bytes(), data)

    def test_copy_res_normalization_and_existing_data_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "hack.res").write_bytes(b"test")
            dest = root / "dest"
            install.copy_game(src, dest, "hack")
            self.assertTrue((dest / "HACK.RES").exists())
            with self.assertRaises(RuntimeError):
                install.copy_game(src, dest, "hack")

    def test_archive_file_selection(self):
        metadata = {"files": [{"name": "EOB3_Disk{}.zip".format(n)} for n in range(1, 5)]}
        self.assertEqual(len(install.archive_choices("eob3", metadata)), 4)
        with self.assertRaises(RuntimeError):
            install.archive_choices("hack", metadata)


if __name__ == "__main__":
    unittest.main()
