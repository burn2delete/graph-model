"""GDM43 measured repair. No candidates were promoted by the original simulator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.measured.run import main
if __name__ == '__main__':
    sys.argv += ['--batches', '43']
    main()
