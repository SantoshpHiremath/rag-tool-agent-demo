"""
test_offline.py
----------------

Manual sanity checks that run without Ollama needing to be active --
confirms the project structure and safe-calculator logic are correct
before running the live demo (which needs a running Ollama instance
with llama3.2 and nomic-embed-text pulled). This mirrors the "offline
first" verification approach used throughout this portfolio: check what
can be checked deterministically before attempting anything that
depends on a live external model.

Run with: python test_offline.py
"""
from pathlib import Path

from calculator_tool import safe_calculate, calculator
from rag_tool import DATA_PATH, _chunk_text


def main():
    print("=== Offline sanity checks ===\n")

    print("1. Data file exists and is readable:")
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    text = DATA_PATH.read_text()
    print(f"   OK -- {len(text)} characters read from {DATA_PATH.name}")

    print("\n2. Chunking produces a reasonable number of non-empty chunks:")
    chunks = _chunk_text(text)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)
    print(f"   OK -- {len(chunks)} chunks")

    print("\n3. Calculator tool is registered correctly:")
    assert calculator.name == "calculator"
    print(f"   OK -- tool name = '{calculator.name}'")

    print("\n4. safe_calculate() arithmetic sanity check:")
    result = safe_calculate("1320 / (3601 + 1320)")
    print(f"   1320 / (3601 + 1320) = {result}")
    assert abs(result - 0.26824) < 0.001

    print("\n5. safe_calculate() rejects code injection:")
    try:
        safe_calculate("__import__('os').system('echo unsafe')")
        raise AssertionError("Expected safe_calculate to raise on injection attempt")
    except AssertionError:
        raise
    except Exception:
        print("   OK -- injection attempt correctly rejected")

    print("\nAll offline sanity checks passed. Project structure and "
          "calculator logic are correct. To run the live demo (needs "
          "Ollama running locally with llama3.2 + nomic-embed-text "
          "pulled): python agent.py")


if __name__ == "__main__":
    main()
