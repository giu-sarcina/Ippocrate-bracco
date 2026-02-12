import os
import json
from datetime import datetime

class RestartManager:
    def __init__(self, local_config):
        self.local_config = local_config
        self.study_path = os.getenv("ROOT_NAME")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_checkfile, exist_ok=True)

    def get_check_phase(self): 
        """
        Determine the latest completed processing phase based on timestamps in available check files.
        In case of identical timestamps, prioritize based on expected processing order:
        input → clinical → image.

        Returns:
            int: The latest completed phase number, or 0 if no valid check files are found.
        """
        filenames = [
            "input_check_file.json",
            "clinical_data_check_file.json",
            "image_data_check_file.json"
        ]
    
        latest_phase = -1
        latest_time = None
        latest_index = -1  # Track order in filenames to break timestamp ties
    
        for idx, filename in enumerate(filenames):
            check_file_path = os.path.join(self.output_directory_checkfile, filename)
    
            if not os.path.exists(check_file_path):
                continue
    
            try:
                with open(check_file_path, "r") as file:
                    data = json.load(file)
                    metadata = data.get("metadata") or data.get("global metadata")
                    timestamp_str = metadata.get("timestamp")
                    phase_number = data.get("phase_number")
    
                    if timestamp_str and phase_number is not None:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        if (
                            latest_time is None or
                            timestamp > latest_time or
                            (timestamp == latest_time and idx > latest_index)
                        ):
                            latest_time = timestamp
                            latest_phase = phase_number
                            latest_index = idx  # Update to file's order index

            except (json.JSONDecodeError, FileNotFoundError, PermissionError, ValueError) as e:
                print(f"Error reading {filename}: {e}")
                continue
    
        return latest_phase if latest_phase != -1 else 0