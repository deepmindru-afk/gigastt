"""Model-free checks for the public-content guard."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check-publication.py").resolve()
SPEC = importlib.util.spec_from_file_location("publication", SCRIPT)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def internal(prefix="TASK", number=123):
    return f"{prefix}-{number}"


class TextTests(unittest.TestCase):
    def test_rejects_identifiers_in_prose_and_branch_names(self):
        for prefix in ("TASK", "TTX", "T", "V1", "SUS", "TODO"):
            for value in (internal(prefix), internal(prefix).lower()):
                for text in (f"Fix {value}", f"fix/resolve-{value}-issue"):
                    with self.subTest(text=text):
                        self.assertTrue(publication.scan_text("metadata", text))

    def test_allows_upstream_specification_and_ordinary_issue_numbers(self):
        text = (
            "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf\n"
            "Fix #123; RFC 6716; release v1.0.0; model v3_rnnt\n"
        )
        self.assertEqual(publication.scan_text("source", text), [])
        self.assertTrue(publication.scan_text("source", text + internal()))


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.hooksPath", str(self.root / "empty-hooks"))
        (self.root / "readme.txt").write_text("Public documentation\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Initial documentation")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root).decode().strip()

    def run_guard(self, *args, input_text=None):
        return subprocess.run(
            ["python3", str(SCRIPT), *args], cwd=self.root,
            input=input_text, text=True, capture_output=True,
        )

    def test_checks_index_instead_of_unstaged_replacement(self):
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        (self.root / "readme.txt").write_text("Clean working copy")
        self.assertNotEqual(self.run_guard().returncode, 0)
        self.git("add", ".")
        self.assertEqual(self.run_guard().returncode, 0)

    def test_private_directories_are_rejected_even_without_identifiers(self):
        for directory in ("backlog", "specs", "roadmap"):
            path = self.root / directory / "notes.md"
            path.parent.mkdir(exist_ok=True)
            path.write_text("Private plan")
            self.git("add", str(path))
            self.assertNotEqual(self.run_guard().returncode, 0)
            self.git("rm", "--cached", str(path))

    def test_branch_and_proposed_commit_message_are_checked(self):
        self.git("checkout", "-qb", "fix/" + internal().lower())
        self.assertNotEqual(self.run_guard().returncode, 0)
        self.git("checkout", "main")
        message = self.root / "message.txt"
        message.write_text("Fix " + internal())
        self.assertNotEqual(self.run_guard("--commit-message-file", str(message)).returncode, 0)

    def test_reverted_intermediate_commit_is_still_checked(self):
        base = self.git("rev-parse", "HEAD")
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Add notes")
        (self.root / "readme.txt").write_text("Public documentation")
        self.git("add", ".")
        self.git("commit", "-qm", "Restore documentation")
        self.assertNotEqual(self.run_guard("--commit-range", base + "..HEAD").returncode, 0)

    def test_commit_body_and_pr_metadata_are_checked(self):
        base = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "Fix behavior\n\n" + internal())
        self.assertNotEqual(self.run_guard("--commit-range", base + "..HEAD").returncode, 0)
        event = self.root / "event.json"
        for field in ("title", "body"):
            payload = {"pull_request": {
                "title": "Public change", "body": "Public description",
                "head": {"ref": "fix/public", "sha": self.git("rev-parse", "HEAD")},
                "base": {"sha": base},
            }}
            payload["pull_request"][field] = internal()
            event.write_text(json.dumps(payload))
            result = self.run_guard("--event-file", str(event))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pull request " + field, result.stderr)

    def test_pre_push_checks_proposed_ref_and_unpublished_commits(self):
        head = self.git("rev-parse", "HEAD")
        ref = "refs/heads/fix/" + internal().lower()
        line = f"{ref} {head} {ref} {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)

    def test_clean_push_and_pr_event_are_accepted(self):
        head = self.git("rev-parse", "HEAD")
        line = f"refs/heads/main {head} refs/heads/main {'0' * 40}\n"
        self.assertEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)
        event = self.root / "event.json"
        event.write_text(json.dumps({"pull_request": {
            "title": "Public change", "body": "Fix #123",
            "head": {"ref": "fix/public", "sha": head}, "base": {"sha": head},
        }}))
        self.assertEqual(self.run_guard("--event-file", str(event)).returncode, 0)

    def test_tag_annotation_and_existing_tip_cannot_bypass_guard(self):
        self.git("tag", "-a", "v1.0.0", "-m", internal())
        tag = self.git("rev-parse", "v1.0.0")
        line = f"refs/tags/v1.0.0 {tag} refs/tags/v1.0.0 {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)
        self.git("checkout", "-qb", "private")
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Private fixture")
        head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/remotes/origin/private", head)
        self.git("checkout", "main")
        line = f"refs/heads/private {head} refs/heads/new {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)

    def test_merge_resolution_cannot_hide_identifier_in_changed_blob(self):
        self.git("checkout", "-qb", "feature")
        (self.root / "readme.txt").write_text("Feature wording")
        self.git("add", ".")
        self.git("commit", "-qm", "Feature documentation")
        self.git("checkout", "main")
        (self.root / "readme.txt").write_text("Main wording")
        self.git("add", ".")
        self.git("commit", "-qm", "Main documentation")
        conflict = subprocess.run(
            ["git", "merge", "feature"], cwd=self.root, capture_output=True,
        )
        self.assertNotEqual(conflict.returncode, 0)
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Resolve documentation")
        # Leave a clean index so detection must inspect the merge itself.
        (self.root / "readme.txt").write_text("Public documentation")
        self.git("add", ".")
        result = self.run_guard("--commit-range", "HEAD^..HEAD")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("readme.txt", result.stderr)


if __name__ == "__main__":
    unittest.main()
