"""Wrapper: analyse and visualise a situations JSONL file."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iri_tfg_program.analysis.study_situations import main
if __name__ == "__main__":
    main()
