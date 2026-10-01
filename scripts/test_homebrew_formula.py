"""Release checksum validation must finish before any formula write."""
import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location("homebrew_formula", Path(__file__).with_name("update-homebrew-formula.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.formula = Path("Formula/gigastt.rb").read_text()
        self.sums = "\n".join(f"{str(i) * 64}  gigastt-2.22.0-{target}.tar.gz" for i, target in enumerate(module.TARGETS, 1))

    def test_rewrite_is_idempotent_and_preserves_non_pin_content(self):
        updated = module.rewrite(self.formula, "v2.22.0", self.sums)
        self.assertEqual(module.rewrite(updated, "v2.22.0", self.sums), updated)
        for i, target in enumerate(module.TARGETS, 1):
            self.assertIn(f'v2.22.0/gigastt-2.22.0-{target}.tar.gz"\n      sha256 "{str(i) * 64}"', updated)
        self.assertEqual(self.formula[self.formula.index("  def install"):], updated[updated.index("  def install"):])

    def test_missing_duplicate_or_invalid_checksums_fail(self):
        for sums in [self.sums.splitlines()[0], self.sums + "\n" + self.sums.splitlines()[0], self.sums.replace("1" * 64, "bad")]:
            with self.subTest(sums=sums), self.assertRaises(ValueError):
                module.rewrite(self.formula, "v2.22.0", sums)

    def test_invalid_asset_leaves_formula_file_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            formula = Path(directory) / "formula.rb"
            sums = Path(directory) / "SHA256SUMS.txt"
            formula.write_text(self.formula)
            sums.write_text(self.sums.replace("3" * 64, "invalid"))
            result = subprocess.run([sys.executable, str(Path(module.__file__)), "v2.22.0", str(sums), "--formula", str(formula)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(formula.read_text(), self.formula)

    def test_invalid_tags_and_unexpected_formula_layout_fail(self):
        for tag in ["main", "v2.22.0\ninjected=x", "v2.22.0/other"]:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                module.rewrite(self.formula, tag, self.sums)
        with self.assertRaises(ValueError):
            module.rewrite(self.formula.replace("  version", "  old_version"), "v2.22.0", self.sums)
        with self.assertRaises(ValueError):
            module.rewrite(self.formula + self.formula, "v2.22.0", self.sums)


if __name__ == "__main__":
    unittest.main()
