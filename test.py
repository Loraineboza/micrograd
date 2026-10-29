import unittest

from tests.test_autograd import TestAutograd


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAutograd)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
