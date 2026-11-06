"""Make the compiled `optlib` module (build/python) importable for the tests."""
import pathlib
import sys

BUILD_PY = pathlib.Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_PY))
