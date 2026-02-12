import numpy as np
import pandas as pd
import glob
import os 

from scipy.stats import f
from typing import List, Dict, Any
from DataManagers import VectorDataset , load_person_series
from torch.utils.data import DataLoader

from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.signal import Signal

from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext

class FIDExecutor(Executor):
    
    def __init__(self, series_name="series001", target_type= None, train_task_name: str = "FID_to_similarity"):
        super().__init__()
        self._train_task_name = train_task_name
        self.series_name=series_name
        self.target_type=target_type
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        pass
    
    def execute(self, task_name:str, shareable: Shareable, fl_ctx: FLContext,abort_signal: Signal) -> Shareable:
        
        self.log_info(fl_ctx, f"Task {task_name} started.")
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            if task_name == self._train_task_name:
                return self.inception_distribution(shareable,fl_ctx, abort_signal)
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}. Exception: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)    
        
    def load_data(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Loading inception features from local dataset ...")
        # 1. Local loading 
        ### DATASET AND DATALOADER ###
        series = load_person_series() #load all image series for each person from omop database
        self.log_info(
            fl_ctx,
            f"Dataset Size: {len(series)}",
        )

        # CUSTOM DATASET 
        self.dataset = VectorDataset(series, series_name=self.series_name,target_type=None)
        self.batch_size= len(self.dataset)  # all data in one batch
        features = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=0
                                )
        return features

    def inception_distribution(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):

        loader = self.load_data(fl_ctx)

        n_total = 0
        mean = None
        M2 = None   # per var/cov online

        for batch in loader:
            batch = batch[0]  
            batch = batch.numpy()               # (B, d)
            B = batch.shape[0]

            if mean is None:
                d = batch.shape[1]
                mean = np.zeros(d)
                M2 = np.zeros((d, d))

            # mean aggiornato
            batch_mean = batch.mean(axis=0)
            delta = batch_mean - mean

            new_n = n_total + B
            mean += delta * (B / new_n)

            # cov update
            # differenze dei singoli elementi dal loro batch mean
            X = batch - batch_mean
            M2 += X.T @ X + np.outer(delta, delta) * (n_total * B / new_n)

            n_total = new_n

        # cov finale
        sigma = M2 / (n_total - 1)
        
        # 5. Prepara output
        output = {
            "stats": {
                "mean": mean,
                "cov": sigma
            }
        }
        
        # 3. DXO to send to server
        data = output
        dxo = DXO(data_kind="WEIGHTS", data=data)
        shareable = dxo.to_shareable()
        shareable.set_header("status", TaskCompletionStatus.OK)
        return shareable


