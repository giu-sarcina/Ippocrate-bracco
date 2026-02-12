import os
import csv
import pandas as pd
import json


def read_csv_file(file_path):
    """
    Reads a CSV file and detects its delimiter.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        tuple: A tuple containing the DataFrame and the detected delimiter.
    
    Raises:
        ValueError: If the file does not exist or the delimiter cannot be detected.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' not found.")

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        sniffer = csv.Sniffer()
        sample = csvfile.read(1024)  # Read a small sample of the file
        csvfile.seek(0)  # Reset file pointer to the beginning
        dialect = sniffer.sniff(sample)

        # Determine the delimiter
        if dialect.delimiter == '\t':
            sep = '\t'
        elif dialect.delimiter == ';':
            sep = ';'
        elif dialect.delimiter == ',':
            sep = ','
        else:
            #raise ValueError("Unsupported delimiter detected in the file.")
            sep=None

        if sep is not None:
            # Read the clinical data file
            data_df = pd.read_csv(file_path, sep=sep)
        else:
            data_df = pd.read_csv(file_path)
    
    return data_df


def read_csv_file_two_layer_label(file_path):
    """
    Reads a CSV file with a two-layer header and detects its delimiter.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        tuple: A tuple containing the DataFrame and the detected delimiter.
    
    Raises:
        ValueError: If the file does not exist or the delimiter cannot be detected.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' not found.")

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        sniffer = csv.Sniffer()
        sample = csvfile.read(1024)  # Read a small sample of the file
        csvfile.seek(0)  # Reset file pointer to the beginning
        dialect = sniffer.sniff(sample)

        # Determine the delimiter
        if dialect.delimiter == '\t':
            sep = '\t'
        elif dialect.delimiter == ';':
            sep = ';'
        elif dialect.delimiter == ',':
            sep = ','
        else:
            #raise ValueError("Unsupported delimiter detected in the file.")
            sep=None

        if sep is not None:
            # Read the clinical data file
            data_df = pd.read_csv(file_path, sep=sep, header=[0, 1])
        else:
            data_df = pd.read_csv(file_path, header=[0, 1])
    
    return data_df


def save_state(state, state_file):
    """
    Save the current state to a JSON file.
    Args:
        state (dict): The state dictionary to save.
        state_file (str): The file path to save the state.
    """
    with open(state_file, "w") as f:
        json.dump(state, f)


def load_state(state_file):
    """
    Load the state from a JSON file if it exists.
    Args:
        state_file (str): The file path from which to load the state.
    Returns:
        dict: The loaded state dictionary, or an empty dictionary if the file doesn't exist.
    """
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {}