"""Wrapper: demo the full situations-to-training-sample pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iri_tfg_program.analysis.demo_situations_pipeline import main
if __name__ == "__main__":
    main()
