# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
#WORKSPACE
from nvflare.apis.workspace import Workspace # FLContextKey.WORKSPACE_OBJECT
#CONSTANTS
from nvflare.apis.event_type import EventType #EventType.END_RUN
from nvflare.apis.fl_constant import FLContextKey # FLContextKey.WORKSPACE_OBJECT 
from nvflare.apis.fl_constant import FLMetaKey # FLMetaKey.CURRENT_ROUND, FLMetaKey.JOB_ID
from nvflare.app_common.app_constant import AppConstants #AppConstants.IS_BEST

import pandas as pd
import numpy as np
import random

import torch
import torchvision
import torchio as tio

from torch import optim 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

from learners.supervised_learner import SupervisedLearner
from PIL import Image

#custom 
from DataManagers import load_person_series, subset_split, TorchIOImageDataset
from unet3d.metrics import DiceCoefficient
from unet3d.losses import DiceLoss
from unet import UNet3D 

#MONAI
#from monai.inferers import SlidingWindowInferer  

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

class Ensure4D(tio.Transform):
    def apply_transform(self, img):
        t = img.data
        if t.ndim == 3:
            t = t.unsqueeze(0)  # aggiunge canale: (1, X, Y, Z)
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(-1)  # (1, X, Y, 1)
        img.set_data(t)
        return img

class SupervisedProstateLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Learner for prostate segmentation task.
        It inherits from SupervisedLearner.

        Args:
            train_config_filename: path for config file, this is an addition term for config loading
            aggregation_epochs: the number of training epochs for a round.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__(
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
        )
        self.train_config_filename = train_config_filename
        self.config_info = None

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        # save the model locally 

        # tensorboard streaming 
        
        pass

    def train_config(self, fl_ctx: FLContext):
        """Traning configuration
        Here, we use a json to specify the needed parameters
        """
        ### LOAD TRAIN CONFIG FILE ###
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()

        # CONFIGURAZIONE
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        # Get the config_info
        ##training hyperparameters
        self.lr = self.config_info.get("learning_rate", 0.001)
        self.batch_size = self.config_info.get('batch_size', 1)
        self.central = self.config_info.get('central', False)
        self.gpu = self.config_info.get('gpu', 'cuda:0')
        ##architecture and input 
        self.unet_input_dim = self.config_info.get("unet_input_dim", 3)
        self.n_unet_blocks=self.config_info.get("n_unet_blocks", 3)
        self.roi_size = self.config_info.get("roi_size", [128, 128, 32])
        self.in_channels = self.config_info.get("in_channels", 1)
        self.out_channels = self.config_info.get("out_channels", 1)
        #series 
        self.series_name = self.config_info['series_name']
        self.target_type = self.config_info['target_type']
        
        ### MODEL, OPTIMIZER, LOSS, INFERER AND EVAL METRIC ###
        self.device = torch.device(self.gpu if torch.cuda.is_available() else "cpu")
        print(f" ############## Using device: {self.device} ################")

        ''' 
        class UNet(nn.Module):
        def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
            ):
        '''
        self.model = UNet3D(
                in_channels=int(self.in_channels),
                out_classes=int(self.out_channels),
                dimensions=int(self.unet_input_dim),
                num_encoding_blocks=int(self.n_unet_blocks),
                out_channels_first_layer=8,
                normalization='batch',
                upsampling_type='linear',
                padding=1,
                activation='PReLU',
                initial_dilation=1,
            ).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = DiceLoss( sigmoid_normalization=True,)
        self.valid_metric = DiceLoss( sigmoid_normalization=True,) #DiceCoefficient()
        
        #self.inferer = SlidingWindowInferer(roi_size=self.infer_roi_size, sw_batch_size=4, overlap=0.25)

        ### TRANSFORMS ###
        self.transform_train = tio.Compose([     
            tio.CropOrPad(tuple(self.roi_size), mask_name = "mask"),  
            tio.RandomFlip(axes=(0,), flip_probability=0.5),   
            tio.RandomFlip(axes=(1,), flip_probability=0.5),  
            tio.ZNormalization(masking_method=lambda x: x > torch.tensor(0)),       
            tio.RandomGamma(log_gamma=(-0.1, 0.1), p=1.0),     
            tio.RandomBiasField(coefficients=0.1, p=1.0),   
            tio.EnsureShapeMultiple(8, method='pad') ,                              
            tio.To(torch.float32)                
        ])
        
        self.transform_valid = tio.Compose([        
            tio.CropOrPad(tuple(self.roi_size), mask_name = "mask"),  
            tio.ZNormalization(masking_method=lambda x: x > torch.tensor(0)),
            tio.EnsureShapeMultiple(8, method='pad')  , # perchè la rete ha 2^3 blocchi di downsampling/upsampling
            tio.To(torch.float32)                 
        ])
       
        # MASK POST PROCESSING
        self.transform_post = tio.Compose([         
            tio.CropOrPad(tuple(self.roi_size), mask_name = "mask"),  
            tio.EnsureShapeMultiple(8, method='pad') ,  
            tio.To(torch.float32)                 
        ])

        ### DATASET AND DATALOADER ###
        series = load_person_series() #load all image series for each person from omop database
        train_split, val_split = subset_split(series, val_ratio=0.2, shuffle=True)
        self.log_info(
            fl_ctx,
            f"Training Size: {len(train_split)}, Validation Size: {len(val_split)}",
        )

        # CUSTOM DATASET 
        self.train_dataset = TorchIOImageDataset(train_split, 
                                        transform=self.transform_train,
                                        series_name=self.series_name,
                                        target_type=self.target_type)
        self.valid_dataset = TorchIOImageDataset(val_split, 
                                        transform=self.transform_valid,
                                        series_name=self.series_name,
                                        target_type=self.target_type)
        ### DATALOADER ###
        self.train_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=1)
        
        self.valid_loader = DataLoader(self.valid_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=False, 
                                    num_workers=1)

    #Giulia : save the final model at the end of training
    def handle_event(
        self,
        event_type: str,
        fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)

        if event_type == EventType.END_RUN:
            ws: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_job_id()
            run_dir = ws.get_run_dir(job_id)  
            
            #current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            
            round_num = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)#FLContextKey.CURRENT_RUN
            save_path = os.path.join(
                run_dir,
                f"model_round_{round_num}.pth"
            )

            torch.save(self.model.state_dict(), save_path)
            self.log_info(fl_ctx, f"Model saved at {save_path}")

            # save best model
            if fl_ctx.get_prop(AppConstants.IS_BEST, False):
                best_save_path = os.path.join(run_dir, f"best_model-round_{round_num}.pth")
                torch.save(self.model.state_dict(), best_save_path)
                self.log_info(fl_ctx, f"Best model updated at {best_save_path}") 