"""GDM65 entrypoint reusing the canonical GDM64 numerical wrapper."""
from experiments.followup64 import execute as e64
from experiments.followup64 import run as r64
from experiments.followup65 import run as r65

def main():
    old=r64.main
    r64.main=r65.main
    try:
        e64.main()
    finally:
        r64.main=old

if __name__=="__main__":
    main()
