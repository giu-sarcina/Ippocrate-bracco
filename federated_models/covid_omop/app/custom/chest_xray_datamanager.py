import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
from pathlib import Path
from scipy.ndimage import zoom
import os
import psycopg2
import random

import logging

# Global counter for incremental image naming
_image_save_counter = 0

POSSIBLE_LABELS = ['Normal', 'COVID']



def load_data_from_OMOP(client_id):
    """
    Load data from OMOP database and create a dataframe with image information.

    Args:
        client_id: The client ID to assign to all rows

    Returns:
        pandas.DataFrame: DataFrame with columns: image_path, seg_path, client, split, label
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
        return pd.DataFrame(columns=['image_path', 'seg_path', 'client', 'split', 'label'])

    try:
        # -------------------------
        # Query execution
        # -------------------------
        cursor = conn.cursor()

        query = """
        SELECT filename, seg_mask, patient_label
        FROM image_occurrence
        WHERE patient_label = ANY(%s);
        """

        logging.error(f"Executing query with parameters: {POSSIBLE_LABELS}")
        cursor.execute(query, (POSSIBLE_LABELS,))
        rows = cursor.fetchall()

        logging.warning(f"Found {len(rows)} rows in image_occurrence table")

        if rows:
            logging.error(f"First row: {rows[0]}")

        # -------------------------
        # Data processing
        # -------------------------
        data = []
        for filename, seg_mask, patient_label in rows:

            if filename is None or seg_mask is None or patient_label is None:
                logging.error(
                    f"Skipping row with None values: "
                    f"{(filename, seg_mask, patient_label)}"
                )
                continue

            split = 'train' if random.random() < 0.8 else 'val'
            label = 1 if patient_label == 'COVID' else 0

            data.append({
                'image_path': filename,
                'seg_path': seg_mask,
                'client': client_id,
                'split': split,
                'label': label
            })

        logging.error(f"Processed {len(data)} valid rows")

        images_info = pd.DataFrame(data)

        logging.error(f"Loaded {len(images_info)} images from OMOP database")

        # -------------------------
        # Save output
        # -------------------------
        try:
            save_path = Path('/home/output/images_info.csv')
            images_info.to_csv(save_path, index=False)
            logging.error(f"Saved images_info dataframe to {save_path}")
        except Exception as e:
            logging.error(f"Warning: Could not save dataframe: {e}")

        return images_info

    except Exception as e:
        logging.error(f"Unexpected error during database operation: {e}")
        import traceback
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        return pd.DataFrame(columns=['image_path', 'seg_path', 'client', 'split', 'label'])

    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def load_and_preprocess_image_png(img_path, mask_path=None, transform=None):
    logging.error('load_and_preprocess_image')
    with open(img_path, 'rb') as file:
        img = Image.open(file)

        # Apply mask if available
        if mask_path is not None:
            try:
                with open(mask_path, 'rb') as mask_file:
                    mask = Image.open(mask_file)
                    mask = mask.resize(img.size, Image.Resampling.NEAREST)
                    # Convert both to same mode (e.g., RGB) if needed
                    if img.mode != mask.mode:
                        mask = mask.convert(img.mode)
                    # Apply mask by multiplying pixel values
                    img = Image.fromarray((np.array(img) * (np.array(mask) / 255.0)).astype(np.uint8))
            except (FileNotFoundError, IOError):
                pass  # Continue with original image if mask can't be loaded

        if transform is not None:
            img = transform(img)
            img = convert_image_dtype(img)
    return img


def load_and_preprocess_image(img_path, mask_path=None, transform=None):
    global _image_save_counter
    logging.error('load_and_preprocess_image')
    
    # Load image as numpy array from .npy file
    img = np.load(img_path)
    
    # Apply mask if available
    if mask_path is not None:
        try:
            # Load mask as numpy array from .npy file
            mask = np.load(mask_path)
            
            # Ensure mask has same shape as image
            if img.shape != mask.shape:
                # Resize mask to match image dimensions if needed
                
                zoom_factors = [img.shape[i] / mask.shape[i] for i in range(len(img.shape))]
                mask = zoom(mask, zoom_factors, order=0)  # order=0 for nearest neighbor interpolation
            
            # Normalize mask to 0-1 range if it's not already
            if mask.max() > 1.0:
                mask = mask / 255.0
            
            # Apply mask by multiplying pixel values
            img = img * mask
            
        except (FileNotFoundError, IOError, ValueError):
            pass  # Continue with original image if mask can't be loaded
    
    # Convert to PIL Image if transform is needed
    if transform is not None:
        # # Handle different array shapes and data types
        # if len(img.shape) == 2:  # Grayscale 2D
        #     img = Image.fromarray(img.astype(np.uint8), mode='L')
        # elif len(img.shape) == 3:  # 3D array
        #     if img.shape[0] == 1 and img.shape[1] == 1:  # Shape like (1, 1, 299)
        #         # Reshape to 1D and then to 2D
        #         img_reshaped = img.flatten().reshape((int(np.sqrt(img.shape[2])), int(np.sqrt(img.shape[2]))))
        #         img = Image.fromarray(img_reshaped.astype(np.uint8), mode='L')
        #     elif img.shape[2] == 1:  # Shape like (H, W, 1)
        #         img = Image.fromarray(img[:, :, 0].astype(np.uint8), mode='L')
        #     elif img.shape[2] == 3:  # RGB
        #         img = Image.fromarray(img.astype(np.uint8))
        #     else:  # Other 3D shapes
        #         # Try to reshape to 2D by taking the first channel
        #         img = Image.fromarray(img[:, :, 0].astype(np.uint8), mode='L')
        # else:  # 1D or other shapes
        #     # Try to reshape to a square 2D image
        #     size = int(np.sqrt(img.size))
        #     if size * size == img.size:
        #         img = Image.fromarray(img.reshape((size, size)).astype(np.uint8), mode='L')
        #     else:
        #         # Fallback: create a simple 2D array
        #         img = Image.fromarray(np.zeros((299, 299), dtype=np.uint8), mode='L')
        img = img.astype(np.uint8)

        # Remove the channel dimension to get shape (299, 299) for grayscale image
        np_image = img[0]  # shape: (299, 299)

        # Convert to PIL Image (grayscale mode 'L')
        pil_image = Image.fromarray(np_image, mode='L')
        
        img = transform(pil_image)
        img = convert_image_dtype(img)
    
    # 1% probability to save the image
    if random.random() < 0.01:
        try:
            # Create the directory if it doesn't exist
            save_dir = Path('/home/output')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate incremental filename
            _image_save_counter += 1
            filename = f"processed_image_{_image_save_counter:06d}.png"
            save_path = save_dir / filename
            
            # Convert tensor back to PIL Image for saving
            if hasattr(img, 'save'):  # Already a PIL Image
                save_img = img
            else:  # Tensor - convert back to PIL Image
                # Denormalize if needed and convert to uint8
                if img.dtype == torch.float32:
                    # Assuming normalization to [0,1] range
                    save_img = (img * 255).clamp(0, 255).to(torch.uint8)
                else:
                    save_img = img.to(torch.uint8)
                
                # Convert to PIL Image
                if len(save_img.shape) == 3:  # CxHxW
                    save_img = save_img.permute(1, 2, 0)  # HxWxC
                
                save_img = Image.fromarray(save_img.cpu().numpy())
            
            # Save the image
            save_img.save(save_path)
            logging.info(f"Saved image with 1% probability: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save image: {e}")
    
    return img



def prepare_data(images_info, transform=None, central=False, client_id=None, dset='train'):
    logging.error('prepare_data')
    
    # Check if images_info is empty or doesn't have required columns
    if images_info.empty or 'split' not in images_info.columns:
        logging.error(f"images_info is empty or missing required columns. Columns: {list(images_info.columns) if not images_info.empty else 'empty'}")
        return np.array([]), np.array([])
    
    #base_path = Path('/home/orobix-bracco_covid_model-d991c1da8cac/images')
    #images_split_info = pd.read_csv('/home/orobix-bracco_covid_model-d991c1da8cac/images_split_info.csv')

    if central:
        split = 'train' if dset == 'train' else 'val'
        df_images = images_info[images_info['split'] == split]
    else:
        df_images = images_info[(images_info['client'] == client_id) & (images_info['split'] == dset)]
    
    # Check if filtered dataframe is empty
    if df_images.empty:
        logging.error(f"No images found for client_id={client_id}, dset={dset}, central={central}")
        return np.array([]), np.array([])
    
     # Get image paths
    image_paths = [img_path for img_path in df_images['image_path']]

    # Try to get corresponding mask paths
    mask_paths = []
    # for img in df_images['paths']:
    #     mask_path = base_path / ('Normal' if 'Normal' in img else 'COVID') / 'masks' / img
    #     mask_paths.append(mask_path if mask_path.exists() else None)

    for seg_path in df_images['seg_path']:
        if seg_path is not None and seg_path != '':
            mask_paths.append(seg_path)
        else:
            mask_paths.append(None)
    
    #mask_paths = [img_path for img_path in df_images['seg_path']]

    # Load images with their masks if available
    images = [load_and_preprocess_image(img_path, mask_path, transform)
             for img_path, mask_path in zip(image_paths, mask_paths)]
    labels = df_images['label']

    images = np.array([np.array(img) for img in images])
    labels = np.array(labels)

    return images, labels




class Xray_Chests_Idx(torch.utils.data.Dataset):
    def __init__(self, client_id=None, transform=None, central=False, dset='train'):
        """X-ray chests dataset with index to extract subset

        Args:
            data_idx: to specify the data for a particular client site,
                      if index list is provided, extract subset, otherwise use the whole set
            transform: image transformer
            alpha: percentile of split to use for the training/validation data (default: 1., only training)
            dset: specigy if 'train' or 'valid' dataset is needed  (default: 'train')
            seed: seed for initialize randomization process
        Returns:
            A PyTorch dataset
        """
        self.client_id = client_id
        self.transform = transform
        self.dset = dset
        self.central = central
        self.data, self.target = self.__build_xchest_subset__()
        self.train_data, self.train_target = self.__build_xchest_subset__()
        self.valid_data, self.valid_target = self.__build_xchest_subset__()

        if self.central:
            print("Pay Attention: you are considering a centralized learning")

    def __build_xchest_subset__(self):
        assert(self.dset in ["train", "val"])
        image_info = load_data_from_OMOP(self.client_id)
        data, target = prepare_data(image_info, transform=self.transform, central=self.central, client_id=self.client_id, dset=self.dset)

        return data, target

    def __getitem__(self, index):
        # Get from single index - use the appropriate data based on dataset type
        if self.dset == 'train':
            img, target = self.train_data[index], self.train_target[index]
        else:  # val
            img, target = self.valid_data[index], self.valid_target[index]
        return img, target

    def __len__(self):
        # Return length based on dataset type
        if self.dset == 'train':
            return len(self.train_data)
        else:  # val
            return len(self.valid_data)

if __name__=='__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform_ = transforms.Compose(
                 [
                  transforms.ToTensor()
                 ]
                )

    #pid=None
    my_train = Xray_Chests_Idx(transform=transform_, central=False, client_id='client1', dset='train')
    #print("TRAIN:\n", my_train.target, my_train.data)
    my_val = Xray_Chests_Idx(transform=transform_, central=False, client_id='client1', dset='val')
    #print("VALID:\n", my_valid.target, my_valid.data)

    train_loader = DataLoader(my_train,
                              batch_size=8,
                              shuffle=True,
                              num_workers=0
                             )
    valid_loader = DataLoader(my_val,
                              batch_size=8,
                              shuffle=False,
                              num_workers=0
                             )
