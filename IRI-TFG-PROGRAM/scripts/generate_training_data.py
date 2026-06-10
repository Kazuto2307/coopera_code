"""Wrapper: generate preference training data (Pipeline A - external situations)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iri_tfg_program.training_data.generate_training_data import main
if __name__ == "__main__":
    main()
