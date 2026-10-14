import os
import sys
import unittest
from pathlib import Path


class TestProjectEnvironment(unittest.TestCase):
    """Test suite verifying environment, structure, and foundational files."""

    def test_python_version(self):
        """Verify that Python version is 3.10 or higher."""
        major, minor = sys.version_info[:2]
        self.assertTrue(
            (major == 3 and minor >= 10),
            f"Expected Python 3.10+, got {major}.{minor}"
        )

    def test_project_structure(self):
        """Verify that all core directories and sentinel files exist."""
        root = Path(__file__).resolve().parent.parent
        expected_dirs = ["data", "notebooks", "src", "models", "reports", "tests"]
        for dir_name in expected_dirs:
            dir_path = root / dir_name
            self.assertTrue(dir_path.is_dir(), f"Expected directory missing: {dir_name}")

    def test_essential_files_exist(self):
        """Verify that foundational configuration and documentation files exist."""
        root = Path(__file__).resolve().parent.parent
        expected_files = [
            "requirements.txt",
            ".gitignore",
            "README.md",
            "data/README.md",
            "src/__init__.py",
        ]
        for rel_path in expected_files:
            file_path = root / rel_path
            self.assertTrue(file_path.is_file(), f"Expected file missing: {rel_path}")

    def test_src_import(self):
        """Verify that src package can be imported and exposes version info."""
        import src
        self.assertTrue(hasattr(src, "__version__"), "src package does not define __version__")


if __name__ == "__main__":
    unittest.main()

