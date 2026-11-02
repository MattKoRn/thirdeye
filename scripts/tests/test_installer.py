"""Dependency-free installer regression tests; no game data or downloads needed."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
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

    def test_saved_bin_is_not_mistaken_for_cd_image(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "installed"
            game.mkdir()
            (game / "EYE.RES").write_bytes(b"sample")
            (game / "SAVEGAME").mkdir()
            (game / "SAVEGAME" / "ITEMS_00.BIN").write_bytes(b"save")
            with patch.object(install, "raw_bin_to_iso",
                              side_effect=AssertionError("save treated as CD")):
                self.assertEqual(install.expand_archives(root, "eob3"), game)

    def test_arj_volumes_are_extracted_together_not_with_7zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(1, 7):
                (root / ("DATA%d.ARJ" % n)).write_bytes(b"ARJ volume")
            extracted = []

            def fake_extract(volumes, destination, game):
                extracted.append((list(volumes), destination, game))
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "EYE.RES").write_bytes(b"game data")

            with patch.object(install, "extract_arj_volumes",
                              side_effect=fake_extract), \
                 patch.object(install, "extract_external",
                              side_effect=AssertionError("7-Zip used for ARJ")):
                found = install.expand_archives(root, "eob3")
            self.assertEqual(found, root / "unpacked_game")
            self.assertEqual(len(extracted), 1)
            volumes, destination, game = extracted[0]
            self.assertEqual([p.name for p in volumes],
                             ["DATA%d.ARJ" % n for n in range(1, 7)])
            self.assertEqual(game, "eob3")

    def test_native_arj_invokes_built_decoder_with_space_in_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "folder with spaces"
            directory.mkdir()
            extractor = directory / "thirdeye-unarj.exe"
            extractor.touch()
            volume = directory / "DATA1.ARJ"
            volume.touch()
            target = directory / "game output"
            with patch.object(install, "unarj_executable", return_value=extractor), \
                 patch.object(install, "run") as run:
                install.extract_arj_volumes([volume], target, "eob3")
                run.assert_called_once_with([extractor, target, "eob3", volume])

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


    def test_winget_installed_uses_exact_id_not_only_exit_code(self):
        from types import SimpleNamespace
        with patch.object(install.subprocess, "run",
                          return_value=SimpleNamespace(
                              returncode=0,
                              stdout="Name  Id  Version\\n7-Zip  7zip.7zip  25.01")) as query:
            self.assertTrue(install.winget_installed("7zip.7zip"))
            query.assert_called_once()
            self.assertEqual(query.call_args.args[0][:4],
                             ["winget", "list", "--id", "7zip.7zip"])
        with patch.object(install.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0,
                                                       stdout="No installed package found")):
            self.assertFalse(install.winget_installed("7zip.7zip"))
        with patch.object(install.subprocess, "run",
                          return_value=SimpleNamespace(returncode=1,
                                                       stdout="7zip.7zip")):
            self.assertFalse(install.winget_installed("7zip.7zip"))
        with patch.object(install.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0,
                                                       stdout="7zip.7zip.fake")):
            self.assertFalse(install.winget_installed("7zip.7zip"))

    def test_winget_installer_skips_existing_7zip(self):
        with patch.object(install, "winget_installed", return_value=True), \
             patch.object(install, "run") as run:
            install.winget_install_if_missing("7zip.7zip")
            run.assert_not_called()

    def test_winget_installer_accepts_verified_already_installed_error(self):
        error = install.subprocess.CalledProcessError(
            2316632107, ["winget", "install", "--id", "7zip.7zip"])
        with patch.object(install, "winget_installed",
                          side_effect=[False, True]) as installed, \
             patch.object(install, "run", side_effect=error) as run:
            install.winget_install_if_missing("7zip.7zip")
            installed.assert_called_with("7zip.7zip")
            run.assert_called_once()

    def test_winget_installer_still_fails_on_real_install_error(self):
        error = install.subprocess.CalledProcessError(
            1, ["winget", "install", "--id", "7zip.7zip"])
        with patch.object(install, "winget_installed", return_value=False), \
             patch.object(install, "run", side_effect=error):
            with self.assertRaises(install.subprocess.CalledProcessError):
                install.winget_install_if_missing("7zip.7zip")

    def test_incremental_build_preserves_objects_and_installed_binary(self):
        """Unchanged reruns skip configure, and clean builds are explicit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            home = Path(tmp) / "home"
            home.mkdir()
            build = root / "build" / "installer"
            calls = []

            def fake_run(command, **_kwargs):
                command = list(map(str, command))
                calls.append(command)
                if "-S" in command:
                    build.mkdir(parents=True, exist_ok=True)
                    (build / "CMakeCache.txt").write_text("configured\n")
                    (build / "build.ninja").write_text("ninja\n")
                elif "--build" in command:
                    # An actual incremental generator would only touch this
                    # when inputs changed; simulate an identical build result.
                    (build / "thirdeye").write_bytes(b"engine")

            with patch.object(install, "ROOT", root), \
                 patch.object(install.shutil, "which", return_value="/usr/bin/cmake"), \
                 patch.object(install, "run", side_effect=fake_run):
                exe = install.build_engine("linux", home)
                self.assertTrue(exe.is_file())
                installed_timestamp = exe.stat().st_mtime_ns
                (build / "kept-object.o").write_bytes(b"object")

                install.build_engine("linux", home)
                self.assertEqual(sum("-S" in cmd for cmd in calls), 1)
                self.assertEqual(sum("--build" in cmd for cmd in calls), 2)
                self.assertTrue((build / "kept-object.o").exists())
                self.assertEqual(exe.stat().st_mtime_ns, installed_timestamp)

                # A settings change requires reconfigure, not object removal.
                stamp = build / ".installer-config.json"
                settings = json.loads(stamp.read_text())
                settings["options"] = []
                stamp.write_text(json.dumps(settings))
                install.build_engine("linux", home)
                self.assertEqual(sum("-S" in cmd for cmd in calls), 2)
                self.assertTrue((build / "kept-object.o").exists())

                install.build_engine("linux", home, clean=True)
                self.assertEqual(sum("-S" in cmd for cmd in calls), 3)
                self.assertFalse((build / "kept-object.o").exists())


if __name__ == "__main__":
    unittest.main()
