"""Bound Windows-reader sharing races without weakening data or process guards."""
from pathlib import Path
import json,time

def save(path,value):
    path=Path(path);stage=path.with_suffix('.stage')
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    stage.write_text(payload)
    for attempt in range(101):
        try:
            stage.replace(path)
            return
        except PermissionError:
            if attempt==100:raise
            time.sleep(.02)
