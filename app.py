"""Compatibility entry point; launch V2 by default."""
from pathlib import Path
import runpy
import sys
if __name__ == '__main__':
    version = Path(__file__).resolve().parent / 'V2'
    sys.path.insert(0, str(version))
    runpy.run_path(str(version / 'app.py'), run_name='__main__')
