"""Wrapper: generate synthetic user profiles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iri_tfg_program.profiles.generate_profiles import main
if __name__ == "__main__":
    main()
