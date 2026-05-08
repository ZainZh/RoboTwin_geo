import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestTrainObjpcResumeInterface(unittest.TestCase):
    def test_train_objpc_exposes_resume_flag(self):
        train_script = (REPO_ROOT / "policy" / "DP3" / "train_objpc.sh").read_text(encoding="utf-8")

        self.assertIn("resume=${6:-true}", train_script)
        self.assertIn("object_placeholders=${7:-\\{A\\},\\{B\\}}", train_script)
        self.assertIn("training.resume=${resume}", train_script)


if __name__ == "__main__":
    unittest.main()
