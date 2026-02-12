import os
import torch
import pandas as pd
import numpy as np
import psycopg2
import logging
import json
from sklearn.model_selection import train_test_split

POSSIBLE_LABELS = ['disease_1', 'disease_2']



def load_data_from_OMOP():
    """
    Load data from OMOP database and create a dataframe with genomic information.

    Args:
    

    Returns:
        Two paths to the genomic data file (training and validation), pandas.DataFrame:
         DataFrame with columns: VCF2matrix path, label
    """
    DB_NAME = os.getenv('POSTGRES_DB', 'omop_ippocrate_demo')
    DB_USER = os.getenv('POSTGRES_USER', 'user1')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    DB_HOST = "ippocratedb_demo"
    DB_PORT = 5432

    logging.error(
        f"Attempting to connect to database: {DB_NAME} "
        f"as user {DB_USER} on host {DB_HOST}:{DB_PORT}"
    )

    logging.error(
        f"Environment variables: POSTGRES_DB={os.getenv('POSTGRES_DB')}, "
        f"POSTGRES_USER={os.getenv('POSTGRES_USER')}, "
        f"POSTGRES_PASSWORD={'***' if os.getenv('POSTGRES_PASSWORD') else 'None'}"
    )

    hosts_to_try = [(DB_HOST, 5432), ("localhost", os.getenv('POSTGRES_PORT', '5432'))]
    conn = None
    last_error = None

    # -------------------------
    # Connection with fallback
    # -------------------------
    for host, port in hosts_to_try:
        try:
            logging.error(
                f"Trying to connect to {DB_NAME} at {host}:{port} as {DB_USER}"
            )

            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=host,
                port=port
            )

            logging.error(
                f"Successfully connected to database {DB_NAME} "
                f"at {host}:{port} as user {DB_USER}"
            )
            break

        except psycopg2.OperationalError as e:
            logging.error(f"Connection failed using host={host}: {e}")
            last_error = e

    if conn is None:
        logging.error("All database connection attempts failed")
        logging.error(f"Last error: {last_error}")
        return pd.DataFrame(columns=['VCF2matrix', 'label'])

    try:
        # -------------------------
        # Query execution
        # -------------------------
        cursor = conn.cursor()
        
        query = """
        SELECT
            gr.VCF2matrix        AS vcf2matrix_path,
            c.concept_name       AS label
        FROM genomic_result        AS gr
        JOIN procedure_occurrence  AS po ON gr.procedure_occurrence_id = po.procedure_occurrence_id
        JOIN person                AS p  ON po.person_id = p.person_id
        JOIN condition_occurrence  AS co ON co.person_id = p.person_id
        JOIN concept               AS c  ON co.condition_concept_id = c.concept_id
        WHERE c.concept_name = ANY(%s);
        """
        
        logging.error(f"Executing query with POSSIBLE_LABELS: {POSSIBLE_LABELS}")
        cursor.execute(query, (POSSIBLE_LABELS,))
        rows = cursor.fetchall()
        
        logging.error(f"Found {len(rows)} rows matching the query")
        
        # Create DataFrame from query results
        df = pd.DataFrame(rows, columns=['VCF2matrix', 'label'])
        
        cursor.close()
        conn.close()
        
        logging.error(f"Loaded {len(df)} genomic samples from OMOP database")
        return df
        
    except psycopg2.Error as e:
        logging.error(f"Error executing query: {e}")
        if conn:
            conn.close()
        return pd.DataFrame(columns=['VCF2matrix', 'label'])
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if conn:
            conn.close()
        return pd.DataFrame(columns=['VCF2matrix', 'label'])


def generate_datasets(df):
    """
    Generate training and validation datasets from OMOP DataFrame.
    
    Args:
        df: pandas.DataFrame with columns 'VCF2matrix' (paths to JSON files) and 'label'
    
    Returns:
        tuple: (training_csv_path, validation_csv_path)
    """
    if df.empty:
        logging.error("Input DataFrame is empty")
        return None, None
    
    if 'VCF2matrix' not in df.columns or 'label' not in df.columns:
        logging.error("DataFrame must contain 'VCF2matrix' and 'label' columns")
        return None, None
    
    logging.error(f"Processing {len(df)} samples")
    
    # Read JSON files and expand into columns
    json_data_list = []
    failed_paths = []
    
    for idx, row in df.iterrows():
        json_path = row['VCF2matrix']
        label = row['label']
        
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Create a row dict with JSON data + label
            row_dict = dict(json_data)
            row_dict['label'] = label
            json_data_list.append(row_dict)
            
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load JSON from {json_path}: {e}")
            failed_paths.append(json_path)
            continue
    
    if not json_data_list:
        logging.error("No valid JSON files could be loaded")
        return None, None
    
    if failed_paths:
        logging.error(f"Failed to load {len(failed_paths)} JSON files")
    
    # Create DataFrame from expanded JSON data
    expanded_df = pd.DataFrame(json_data_list)
    
    logging.error(f"Expanded DataFrame has {len(expanded_df)} rows and {len(expanded_df.columns)} columns")
    logging.error(f"Columns: {list(expanded_df.columns)}")
    
    # Ensure 'label' column exists
    if 'label' not in expanded_df.columns:
        logging.error("'label' column missing after JSON expansion")
        return None, None
    
    # Convert labels from 'disease_1'/'disease_2' to 0/1
    label_mapping = {'disease_1': 0, 'disease_2': 1}
    expanded_df['label'] = expanded_df['label'].map(label_mapping)
    
    # Check if mapping was successful (no NaN values)
    if expanded_df['label'].isna().any():
        logging.error(f"Found unmapped labels. Unique labels: {expanded_df['label'].unique()}")
        # Drop rows with unmapped labels
        expanded_df = expanded_df.dropna(subset=['label'])
        if expanded_df.empty:
            logging.error("No valid labels after mapping")
            return None, None
    
    # Convert label column to integer
    expanded_df['label'] = expanded_df['label'].astype(int)
    
    # Split into training and validation with balanced validation set
    # Use stratified split to ensure balanced classes in validation set
    train_df, val_df = train_test_split(
        expanded_df,
        test_size=0.2,
        stratify=expanded_df['label'],
        random_state=42
    )
    
    logging.error(f"Training set: {len(train_df)} samples")
    logging.error(f"Validation set: {len(val_df)} samples")
    logging.error(f"Training label distribution:\n{train_df['label'].value_counts()}")
    logging.error(f"Validation label distribution:\n{val_df['label'].value_counts()}")
    
    # Create output directory if it doesn't exist
    output_dir = "/home/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    train_path = "/home/data/training_data_genomic_client.csv"
    val_path = "/home/data/validation_data_genomic_client.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logging.error(f"Saved training data to {train_path}")
    logging.error(f"Saved validation data to {val_path}")
    
    return train_path, val_path

def generate_datasets_from_OMOP():
    df = load_data_from_OMOP()
    return generate_datasets(df)


class GenomicDataset(torch.utils.data.Dataset):
    def __init__(self, data=None, labels=None, data_file=None):
        if data is None and labels is None:
            df = pd.read_csv(data_file)
            # Extract labels (pseudo_id and label columns)
            if "pseudo_id" in df.columns:
                df = df.drop('pseudo_id', axis=1)
            labels_df = df[[ 'label']]
            # Extract data (all columns including pseudo_id, except label)
            data_df = df.drop('label', axis=1)
            # Convert to float32 to match model weights
            self.data = data_df.to_numpy().astype(np.float32)
            self.labels = labels_df.to_numpy().astype(np.float32)
        else:
            self.data = data.cpu().numpy().astype(np.float32)
            self.labels = labels.cpu().numpy().astype(np.float32)

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
if __name__ == "__main__":

    SEED = 0
    ALPHA = 0.8

    data_csv = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-wide.csv")
    labels_csv = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-labels.csv")

    # Read files
    feutures_df = pd.read_csv(data_csv, index_col=0) 
    labels_df = pd.read_csv(labels_csv, index_col=0) 

    # Extract all feutures and labels for the train/valid splitting
    features = features_df.values
    labels = labels_df.values.flatten()

    # Convert to Tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Shuffle
    torch.manual_seed(SEED)
    num_samples = features_tensor.shape[0]
    indices = torch.randperm(num_samples)

    # Apply shuffle
    shuffled_features = features_tensor[indices]
    shuffled_labels = labels_tensor[indices]

    # Split SIZE
    train_size = int(ALPHA * num_samples)
    #valid_size = num_samples - train_size

    # Split Data
    X_train = shuffled_features[:train_size]
    y_train = shuffled_labels[:train_size]
    X_valid = shuffled_features[train_size:]
    y_valid = shuffled_labels[train_size:]

    # Create TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
