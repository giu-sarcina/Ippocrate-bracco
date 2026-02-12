import os
from datetime import datetime
import traceback
import re

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj
    

def log_error(input_dir, phase_name, exception, log_filename):
    os.makedirs(input_dir, exist_ok=True)
    log_file = os.path.join(input_dir, log_filename)

    with open(log_file, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Phase: {phase_name}\n")
        f.write(f"Exception Type: {type(exception).__name__}\n")
        f.write(f"Exception Message: {str(exception)}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write(f"{'='*60}\n")


def clear_log_file(input_dir, log_filename):
    log_file = os.path.join(input_dir, log_filename)
    if os.path.exists(log_file):
        os.remove(log_file)


def numeric_suffix_sort_key(text):
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else float('inf')  # fallback to 'infinite' if no number found


def normalize_path(p: str) -> str: 
    """
    Normalize a path string so that:
    - 'patient\\sample'
    - 'patient/sample'
    become the same key on all OSes.
    """
    return os.path.normpath(p.strip())

    
