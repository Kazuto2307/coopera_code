"""Wrapper: build time-of-day / hourly situations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iri_tfg_program.situations.build_hourly import main
if __name__ == "__main__":
    main()
