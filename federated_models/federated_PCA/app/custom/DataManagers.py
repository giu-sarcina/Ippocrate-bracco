
from operator import index
import os
import psycopg2
import random
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Get connection params from environment
'''
config = dict(DB_NAME=os.getenv('POSTGRES_DB'), 
            DB_USER=os.getenv('POSTGRES_USER'),
            DB_PASSWORD=os.getenv('POSTGRES_PASSWORD'),
            DB_HOST=os.getenv('POSTGRES_HOST'),
            DB_PORT=os.getenv('POSTGRES_PORT'))
'''

DB_NAME = 'omop4ippocrate'
DB_USER = 'giuliamaria'
DB_PASSWORD = 'password'
DB_HOST = "ippocratedb"
#DB_HOST = "omop6postgres14"
DB_PORT =  '5432'

config = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": DB_PORT
}

seed=42
random.seed(seed)
'''
class MyDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = load_image(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img
'''
# tables 
def load_person_clinical_data():
    """
    Load data from OMOP database.
    Returns:
        : tuple with person_id, person series data as list of arrays
        
    logging.info(f"Attempting to connect to database: {os.getenv('POSTGRES_DB')} at {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')} as user {os.getenv('POSTGRES_USER')}")
    logging.info(f"Environment variables: POSTGRES_DB={os.getenv('POSTGRES_DB')}, POSTGRES_USER={os.getenv('POSTGRES_USER')}, POSTGRES_PASSWORD={'***' if os.getenv('POSTGRES_PASSWORD') else 'None'}")
    """
    logging.info(f"Attempting to connect to database: {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
    #logging.info(f"Environment variables: POSTGRES_DB={os.getenv('POSTGRES_DB')}, POSTGRES_USER={os.getenv('POSTGRES_USER')}, POSTGRES_PASSWORD={'***' if os.getenv('POSTGRES_PASSWORD') else 'None'}")
    
    try:
        
        conn = psycopg2.connect(**config)
        #cursor = connection.cursor()
        
        logging.debug(f"Successfully connected to database {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
        
        # Query to select persons with all associated clinical records
        person_clinical_data_query ="""
            WITH combined AS (
            SELECT
                person_id,
                measurement_concept_id AS concept_id,
                value_source_value AS source_value
            FROM measurement
        )
        SELECT
            c.person_id,
            c.concept_id,
            c.source_value
        FROM combined c
        JOIN person p ON c.person_id = p.person_id;

        """
        
        '''
        UNION ALL

            SELECT
                person_id,
                observation_concept_id AS concept_id,
                value_as_number AS source_value
            FROM observation
        '''
        df = pd.read_sql(person_clinical_data_query, conn)
        pivot = df.pivot_table(index="person_id", columns="concept_id", values="source_value", aggfunc="first") # aggfunc="first" se ci sono concept_id duplicati tiene il primo 
        pivot = pivot.reset_index(drop = True)

        logging.info(f"Executing query to load person clinical data")
        conn.close()
        
        logging.info(f"Loaded {len(pivot)} persons from OMOP database")

        return pivot , pivot.columns.tolist()
        
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection error: {e}")
        logging.debug(f"Trying to connect to {DB_HOST}:{DB_PORT} with database {DB_NAME}")
        return exit(1)
    except Exception as e:
        logging.error(f"Unexpected error connecting to database: {e}")
        logging.debug(f"Error type: {type(e)}")
        import traceback
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return exit(1)
# images 
def load_person_series():# ok
    """
    Load data from OMOP database.
    Returns:
        : tuple with person_id, person series data as list of arrays
    """
    logging.info(f"Attempting to connect to database: {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
    logging.info(f"Environment variables: POSTGRES_DB={os.getenv('POSTGRES_DB')}, POSTGRES_USER={os.getenv('POSTGRES_USER')}, POSTGRES_PASSWORD={'***' if os.getenv('POSTGRES_PASSWORD') else 'None'}")

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logging.debug(f"Successfully connected to database {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
        
        # Create cursor
        cursor = conn.cursor()
        
        # Query to select persons with all associated series data as array
        person_series_query = """
            SELECT 
            p.person_id,
            array_agg(ROW(
                i.filename,
                i.seg_mask,
                i.patient_label,
                i.highlights,
                i.note
            )) AS series
        FROM image_occurrence i
        JOIN procedure_occurrence p
        ON i.procedure_occurrence_id = p.procedure_occurrence_id
        GROUP BY p.person_id;
        """
        logging.info(f"Executing query: {person_series_query}")
        cursor.execute(person_series_query)
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        logging.info(f"Loaded {len(rows)} persons from OMOP database")

        return rows
        
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection error: {e}")
        logging.debug(f"Trying to connect to {DB_HOST}:{DB_PORT} with database {DB_NAME}")
        return exit(1)
    except Exception as e:
        logging.error(f"Unexpected error connecting to database: {e}")
        logging.debug(f"Error type: {type(e)}")
        import traceback
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return exit(1)

def subset_split(data, val_ratio=0.2, shuffle=True): #ok
    if shuffle:
        random.shuffle(data)

    split_point = int(len(data) * (1 - val_ratio))

    train = data[:split_point]
    val = data[split_point:]

    return train, val

def select_series_by_series_name(data, series_name):
    for person_id, series_list in data:
        for (filename, seg_mask, feature_vector, patient_label, highlights, note) in series_list:
            if filename.endswith(f"{series_name}.npy"):
                # trovato il record giusto
                return {
                    "filename": filename,
                    "seg_mask": seg_mask,
                    "feature_vector": feature_vector,
                    "patient_label": patient_label,
                    "highlights": highlights,
                    "note": note
                }
            
    return None

def select_series(data, series_name, target_type=None):
    r"""Select series by series_name from data."""
    record = select_series_by_series_name(data, series_name)
    if record is None:
        raise ValueError(f"Series name {series_name} not found in data.")
    
    series = record["filename"]
    seg = record["seg_mask"]
    label = record["patient_label"]
    slice_label = record["highlights"]

    # Normalize target_type into set:
    # - if None → empty set 
    # - if stringa → one element set
    # - if list/tuple → value set 
    if target_type is None:
        targets = set()
    elif isinstance(target_type, str):
        targets = {target_type}
    else:
        targets = set(target_type)

    # define the result targets
    result = []
    if "label" in targets:
        result.append(label)
    if "seg" in targets:
        result.append(seg)
    if "slice_label" in targets:
        result.append(slice_label)

    # return None target
    if not result:
        return series, None
    
    # return one target (series, target)
    if len(result) == 1:
        return series, result[0]

    # return tuple (series, (label, seg, ...))
    return series, tuple(result)
# image vectors 
def select_features(data, series_name):
    r"""Select series by series_name from data."""
    record = select_series_by_series_name(data, series_name)
    if record is None:
        raise ValueError(f"Series name {series_name} not found in data.")
    
    return record["feature_vector"] 

class ClinicalDataset():
    def __init__(self, val_ratio = 0.2, test_ratio=0.1):
        
        r""" Clinical Dataset class to handle clinical data loading and splitting.
        Args:
            val_ratio (float): Proportion of data to use for validation set.
            test_ratio (float): Proportion of data to use for test set.

        """
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        #load series data and target
        self.df_train , self.df_val , self.df_test = self.__build_clinical_dataset__()

    def __build_clinical_dataset__(self):
        self.data, self.columns = load_person_clinical_data()
        
        if self.test_ratio is None:
            self.df_train = self.data
            self.df_test = None
        else:
            self.df_train, self.df_test = train_test_split(
                self.data, test_size=self.test_ratio, shuffle=True, random_state=seed
            )

        if self.val_ratio is None:
            self.df_val = None
        else:
            self.df_train, self.df_val = train_test_split(
                self.df_train, test_size=self.val_ratio, shuffle=True, random_state=seed
            )
            
        return self.df_train, self.df_val, self.df_test
        
    def get_splits(self):
        return self.df_train, self.df_val, self.df_test

class ImageDataset(Dataset):
    def __init__(self, data, transform=None, series_name="series001",target_type=None):
        
        r"""Subclass of class Dataset.
        Map from keys to data samples
        Args:
            data: input data
            transform: image transformer
            series_name: name of the series to load as string
            target_type: "label", "seg", "slice_label" or any combination as list or tuple or None (default)
        Returns:
            A PyTorch Dataset of Tensors
        """
        self.data = data
        self.transform = transform
        self.series_name=series_name
        self.target_type=target_type
        #load series data and target
        self.data, self.target = self.__build_series_dataset__()

    def __build_series_dataset__(self):
        return select_series(self.data, self.series_name, self.target_type)
    
    def __getitem__(self, index):
        # Load image
        filename = self.data[index]
        img = np.load(filename)
        if self.transform:
            img = self.transform(img)

        # None target
        if self.target is None:
            return img

        # Load target(s)
        target_values = []
        targets = self.target
        if not isinstance(targets, (tuple, list)):
            targets = (targets,)

        for t in targets:
            if isinstance(t, str) and t.endswith(".npy"):
                val = np.load(t)                                # array numpy 
            elif isinstance(t, str) and t.endswith(".txt"):
                with open(t, "r") as f:
                    val = f.read().strip()                      # stringa 
            elif isinstance(t, str) and t.endswith(".csv"):
                val = pd.read_csv(t).values  # o list(csv.reader(f))        # array numpy
            else:
                val = t  
            target_values.append(val)

        if len(target_values) == 1:
            target_values = target_values[0]

        return img, target_values
    
    def __len__(self):
        return len(self.data)

class VectorDataset(Dataset):

    def __init__(self, data, series_name="series001"):
        
        r"""Subclass of class Dataset.
        Map from keys to vector samples
        Args:
            data: input data
            series_name: name of the series to load as string
        Returns:
            A PyTorch Dataset of numpy array
        """
        self.data = data
        self.series_name=series_name
        #load series data and target
        self.data = self.__build_vector_dataset__()

    def __build_vector_dataset__(self):
        return select_features(self.data, self.series_name)
    
    def __getitem__(self, index):
        # Load image
        feature_vector_path = self.data[index]
        features = np.load(feature_vector_path)
        return features
    
    def __len__(self):
        return len(self.data) 
    
def main(batch_size=8):
    '''
    series = load_person_series() #load all image series for each person from omop database
    train_split, val_split = subset_split(series, val_ratio=0.2, shuffle=True)

    my_train = vector_Dataset(train_split,  series_name="series001")
    my_val = vector_Dataset(val_split,  series_name="series001")

    train_loader = DataLoader(my_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0
                             )
    val_loader = DataLoader(my_val,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0
                             )
    return train_loader, val_loader 
    '''
    transform_ = transforms.Compose([transforms.ToTensor()])
    series = load_person_series() #load all image series for each person from omop database
    train_split, val_split = subset_split(series, val_ratio=0.2, shuffle=True)

    my_train = imgs_Dataset(train_split, transform=transform_, series_name="series001",target_type="label")
    my_val = imgs_Dataset(val_split, transform=transform_, series_name="series001",target_type="label")
    
    train_loader = DataLoader(my_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0
                             )
    val_loader = DataLoader(my_val,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0
                             )
    return train_loader, val_loader

if __name__=='__main__':
    train_loader, val_loader = main(batch_size=16)    