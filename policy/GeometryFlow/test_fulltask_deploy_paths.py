import unittest

from .deploy_fulltask_tagrt_v2 import _path_list


class FullTaskDeployPathsTest(unittest.TestCase):
    def test_accepts_comma_string_and_sequence(self):
        self.assertEqual(_path_list("a,b"), ["a", "b"])
        self.assertEqual(_path_list(["a", "b"]), ["a", "b"])
        self.assertEqual(_path_list(None), [])


if __name__ == "__main__":
    unittest.main()
