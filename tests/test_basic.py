import unittest

from hask import *

class TestCase(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1, 1)

    def test_infix(self):
        self.assertEqual(3, (identity /icompose2/ add)(1, 2))
        compose2(identity, add)(1, 2.5)

if __name__ == '__main__':
    unittest.main()
