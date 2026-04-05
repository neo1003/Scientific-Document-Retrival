from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_pipeline import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mode" not in argv:
        argv = ["--mode", "query", *argv]
    raise SystemExit(main(argv))
