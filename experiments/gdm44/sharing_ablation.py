"""GDM44 measured repair. Real task-local heads/adapters over frozen encoders.
Independent backbone fine-tuning remains unmeasured; never infer it from this screen.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.measured.run import main
if __name__ == '__main__':
    sys.argv += ['--batches', '44']
    main()
