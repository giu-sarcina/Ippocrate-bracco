
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
from pathlib import Path
import random
import os
import logging

# Global counter for incremental image naming
_image_save_counter = 0

# def load_and_preprocess_image(img_path, transform=None):
#     with open(img_path, 'rb') as file:
#         img = Image.open(file)
#         if transform is not None:
#             img = transform(img)
#             img = convert_image_dtype(img)
#     return img

# def prepare_data(transform=None, central=False, client_id=None, dset='train'):
#     base_path = Path('/tmp/orobix/images')
#     images_split_info = pd.read_csv('/tmp/orobix/images_split_info.csv')

#     if central:
#         split = 'train' if dset == 'train' else 'val'
#         df_images = images_split_info[images_split_info['split'] == split]
#     else:
#         df_images = images_split_info[(images_split_info['client'] == client_id) & (images_split_info['split'] == dset)]

#     image_paths = [base_path / ('Normal' if 'Normal' in img else 'COVID') / 'images' / img for img in df_images['paths']]
#     image_paths = [base_path / img for img in df_images['paths']]
#     images = [load_and_preprocess_image(img_path, transform) for img_path in image_paths]
#     labels = df_images['label']

#     images = np.array([np.array(img) for img in images])
#     labels = np.array(labels)

#     return images, labels


import logging


def load_and_preprocess_image(img_path, mask_path=None, transform=None):
    global _image_save_counter
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
    
    # 1% probability to save the image
    if random.random() < 0.01:
        try:
            # Create the directory if it doesn't exist
            save_dir = Path('/home/federation_sami')
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



def prepare_data(transform=None, central=False, client_id=None, dset='train'):
    logging.error('prepare_data')
    base_path = Path('/home/orobix-bracco_covid_model-d991c1da8cac/images')
    images_split_info = pd.read_csv('/home/orobix-bracco_covid_model-d991c1da8cac/images_split_info.csv')

    if central:
        split = 'train' if dset == 'train' else 'val'
        df_images = images_split_info[images_split_info['split'] == split]
    else:
        df_images = images_split_info[(images_split_info['client'] == client_id) & (images_split_info['split'] == dset)]
     # Get image paths
    image_paths = [base_path / ('Normal' if 'Normal' in img else 'COVID') / 'images' / img for img in df_images['paths']]

    # Try to get corresponding mask paths
    mask_paths = []
    for img in df_images['paths']:
        mask_path = base_path / ('Normal' if 'Normal' in img else 'COVID') / 'masks' / img
        mask_paths.append(mask_path if mask_path.exists() else None)

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
        data, target = prepare_data(transform=self.transform, central=self.central, client_id=self.client_id, dset=self.dset)

        return data, target

    def __getitem__(self, index):
        # Get from single index
        img, target = self.train_data[index], self.train_target[index]
        return img, target

    def __len__(self):
        return len(self.data)

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