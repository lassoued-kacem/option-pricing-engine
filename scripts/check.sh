#!/usr/bin/env bash
# Build + test gate: configure, build, run the C++ and Python test suites.
# Exit 0 = OK (or nothing to check yet). Exit 2 = failure, with the full log
# on stderr.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 0

# Nothing to check yet?
[ -f CMakeLists.txt ] || exit 0
shopt -s nullglob
TESTS=(tests/test_*.py tests/cpp/*.cpp)
[ ${#TESTS[@]} -gt 0 ] || exit 0
PY_TESTS=(tests/test_*.py)

PYTHON=python3
PYTEST="python3 -m pytest"
if [ -x .venv/bin/python ]; then
    PYTHON="$ROOT/.venv/bin/python"
    PYTEST="$PYTHON -m pytest"
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

run() {
    {
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PYTHON" &&
        cmake --build build -j &&
        ctest --test-dir build --output-on-failure &&
        if [ ${#PY_TESTS[@]} -gt 0 ]; then $PYTEST -q; fi
    } >"$LOG" 2>&1
}

if run; then
    tail -n 3 "$LOG"
    exit 0
fi
echo "check.sh: build or tests FAILED" >&2
cat "$LOG" >&2
exit 2
