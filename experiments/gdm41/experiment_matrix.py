"""GDM41 measured repair entrypoint. Original invented rankings are invalid."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.measured.run import main
if __name__ == '__main__':
    sys.argv += ['--batches', '41']
    main()
