"""Compatibility test entry point; delegates to the maintained pytest suite."""

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q"]))
