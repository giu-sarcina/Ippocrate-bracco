
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
import torch
import torchvision
import torchio as tio

from torch.utils.data import DataLoader
from pathlib import Path

# Get connection params from environment
DB_NAME=os.getenv('POSTGRES_DB')
DB_USER=os.getenv('POSTGRES_USER')
DB_PASSWORD=os.getenv('POSTGRES_PASSWORD')
DB_HOST=os.getenv('POSTGRES_HOST')
DB_PORT=os.getenv('POSTGRES_PORT')

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
        tuple with person_id, person series data as list of arrays
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
                jsonb_agg(
                    jsonb_build_object(
                        'filename', i.filename,
                        'seg_mask', i.seg_mask,
                        'feature_vector', i.feature_vector,
                        'patient_label', i.patient_label,
                        'highlights', i.highlights,
                        'note', i.note
                    )
                ) AS series
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
    r"""Select series by series_name from data."""
    patient_table = []
    for person_id, series_list in data:
        for el in series_list:
            if series_name in el["filename"]:
                patient_table.append((
                    el["filename"],
                    el["seg_mask"],
                    el["patient_label"],
                    el["highlights"],
                    el["note"]
                ))
    
    return patient_table if patient_table else None

def select_series_by_series_name(data, series_name):
    r"""Select series by series_name from data."""
    patient_table = []

    # Caso 1: data = (person_id, [series_list])
    if isinstance(data, tuple) and len(data) == 2:
        _, series_list = data
        for el in series_list:
            if series_name in el["filename"]:
                patient_table.append(el)
        return patient_table if patient_table else None

    # Caso 2: data = lista di record [(person_id, [series_list]), ...]
    if isinstance(data, list):
        for person_id, series_list in data:
            for el in series_list:
                if series_name in el["filename"]:
                    patient_table.append(el)
        return patient_table if patient_table else None

    # Caso 3: formato non riconosciuto → errore chiaro
    raise TypeError(f"select_series_by_series_name: formato data non valido: {type(data)} → {data}")

def select_series(data, series_name, target_type=None):
    r"""Select series by series_name from data."""
    record = select_series_by_series_name(data, series_name)
    
    if record is None:
        raise ValueError(f"Series name {series_name} not found in data.")
    
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
    
    series_list = []
    result_list = []

    for el in record:
        series = el["filename"]
        seg = el["seg_mask"]
        label = el["patient_label"]
        slice_label = el["highlights"]

        # define the result targets
        result = []
        if "label" in targets:
            result.append(label)
        if "seg" in targets:
            result.append(seg)
        if "slice_label" in targets:
            result.append(slice_label)

        series_list.append((series, tuple(result) if len(result) > 1 else result[0] if result else None))

    return series_list

def select_features(data, series_name, target_type=None):
    r"""Select series by series_name from data."""
    record = select_series_by_series_name(data, series_name)
    
    if record is None:
        raise ValueError(f"Series name {series_name} not found in data.")
    
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
    
    series_list = []
    result_list = []

    for el in record:
        series = el["feature_vector"]
        seg = el["seg_mask"]
        label = el["patient_label"]
        slice_label = el["highlights"]

        # define the result targets
        result = []
        if "label" in targets:
            result.append(label)
        if "seg" in targets:
            result.append(seg)
        if "slice_label" in targets:
            result.append(slice_label)

        series_list.append((series, tuple(result) if len(result) > 1 else result[0] if result else None))

    return series_list
# image vectors 
def select_features(data, series_name):
    r"""Select series by series_name from data."""
    record = select_series_by_series_name(data, series_name)
    if record is None:
        raise ValueError(f"Series name {series_name} not found in data.")
    
    return record["feature_vector"] 

def fix_shape(img):
    # --- Fix shape ---
    # Shape (H, W, D) 3D
    if img.ndim == 3:
        img = img.unsqueeze(0)                  # -> (1, H, W, D)
        img = img.transpose(1, -1)         # -> (1, H, W, D)

    # shape (H, W) 2D
    elif img.ndim == 2:
        img = img.unsqueeze(0) 
        img = img.unsqueeze(-1)      # -> (1, H, W, 1)
    return img                   

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
    def __init__(self, data, intensity_transform=None, spatial_transform=None, series_name="series001",target_type=None):
        
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
        self.intensity_transform=intensity_transform
        self.spatial_transform=spatial_transform
        self.series_name=series_name
        self.target_type=target_type
        #load series data and target
        #self.data_batch, self.target = self.__build_series_dataset__()

    def __build_series_dataset__(self):
        return select_series(self.data, self.series_name, self.target_type)
    
    def __getitem__(self, index):
        # Load image
        data_batch = self.data[index]
        #data_batch : list of tuples (series, target)
        data_batch = select_series(data_batch, self.series_name, self.target_type)
        batch = []
        for el in data_batch:
            filename = el[0]
            img = np.load(filename)
            
            if self.transform:
                img = self.transform(img)
            if self.spatial_transform:
                img = self.spatial_transform(img)

            target = el[1]
            # None target
            if target is None:
                batch.append(img)
                continue

            # Load target(s)
            target_values = []
            if not isinstance(target, (tuple, list)):
                target = (target,)

            for t in target:
                if isinstance(t, str) and t.endswith(".npy"): # array numpy #segmentation mask
                    val = np.load(t)
                    if self.spatial_transform:
                        val = self.spatial_transform(val)   
                    val = fix_shape(val)                           
                elif isinstance(t, str) and t.endswith(".txt"):  # string # label 
                    with open(t, "r") as f:
                        val = f.read().strip()                      # stringa 
                elif isinstance(t, str) and t.endswith(".csv"): # slice labels
                    val = pd.read_csv(t).values  
                else:
                    val = t  
                target_values.append(val)

            if len(target_values) == 1:
                target_values = target_values[0]
            
            batch.append((img, target_values))
        
        return batch
    
    def __len__(self):
        return len(self.data)
    
class TorchIOImageDataset(Dataset):
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
        self.transform=transform
        self.series_name=series_name
        self.target_type=target_type
        #load series data and target
        #self.data_batch, self.target = self.__build_series_dataset__()

    def __build_series_dataset__(self):
        return select_series(self.data, self.series_name, self.target_type)
    
    def __getitem__(self, index):
        # Load image
        data_batch = self.data[index]
        #data_batch : list of tuples (series, target)
        data_batch = select_series(data_batch, self.series_name, self.target_type)
        batch = []
        for el in data_batch:
            filename = el[0]
            img = np.load(filename) 
            img = torch.from_numpy(img)
            img = fix_shape(img)
            img = tio.ScalarImage(tensor=img)
            
            # torchIO Subject
            fields = {"image": img}
    
            target = el[1]
            # None target
            if target is None:
                batch.append(img)
                continue

            # Load target(s)
            target_values = []
            if not isinstance(target, (tuple, list)):
                target = (target,)

            for t in target:
                if isinstance(t, str) and t.endswith(".npy"): # array numpy #segmentation mask
                    val_array = np.load(t) 
                    val_array = torch.from_numpy(val_array)
                    val_array = fix_shape(val_array)
                    #val = tio.LabelMap(tensor=val_array)
                    # torchIO Subject
                    fields["mask"] = tio.LabelMap(tensor=val_array)
                                   
                elif isinstance(t, str) and t.endswith(".txt"):  # string # label 
                    with open(t, "r") as f:
                        val = f.read().strip()                      # stringa 
                        target_values.append(val)
                elif isinstance(t, str) and t.endswith(".csv"): # slice labels
                    val = pd.read_csv(t).values 
                    target_values.append(val) 
                else:
                    val = t  
            
            subject = tio.Subject(**fields)
            if self.transform:
                subject = self.transform(subject)
            
            img = subject['image'].data
            if "mask" in subject:
                target_values.append(subject['mask'].data)

            if len(target_values) == 1:
                target_values = target_values[0]
            
            batch.append((img, target_values))
        return batch
    
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
        self.data = data # persons 
        self.series_name = series_name # series to load
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