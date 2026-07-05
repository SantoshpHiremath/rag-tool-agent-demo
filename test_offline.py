"""
test_offline.py
----------------
Sanity checks that don't require Ollama to be running - run this first to
confirm the project structure and dependencies are correct before running
the live demo (which needs the Ollama app/service running in the background).

Usage: python test_offline.py
"""

import sys
from pathlib import Path


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    all_ok = True

    all_ok &= check(
        "data/forda_dataset_notes.md exists",
        (Path(__file__).parent / "data" / "forda_dataset_notes.md").exists(),
    )

    try:
        from calculator_tool import safe_calculate
        result = safe_calculate("1320 / (3601 + 1320)")
        all_ok &= check(
            f"calculator_tool.safe_calculate works (1320/(3601+1320) = {result:.4f})",
            abs(result - 0.2683) < 0.001,
        )
    except ImportError as exc:
        all_ok &= check(f"calculator_tool imports (missing dependency: {exc})", False)

    try:
        from calculator_tool import safe_calculate
        blocked = False
        try:
            safe_calculate("__import__('os').system('echo unsafe')")
        except Exception:
            blocked = True
        all_ok &= check("calculator_tool blocks unsafe expressions", blocked)
    except ImportError:
        pass

    try:
        import rag_tool  # noqa: F401
        all_ok &= check("rag_tool module imports cleanly", True)
    except ImportError as exc:
        all_ok &= check(f"rag_tool imports (missing dependency: {exc} - run pip install -r requirements.txt)", False)

    try:
        import agent  # noqa: F401
        all_ok &= check("agent module imports cleanly", True)
    except ImportError as exc:
        all_ok &= check(f"agent imports (missing dependency: {exc})", False)

    print()
    if all_ok:
        print("All offline checks passed. Make sure the Ollama app is running, then run `python agent.py` for the live demo.")
    else:
        print("Some checks failed - run `pip install -r requirements.txt` and re-check.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
