import numpy as np
import pandas as pd

from scipy.stats import f
from typing import List, Dict, Any
from DataManagers import ClinicalDataset


from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.signal import Signal

from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext

def hotelling_t2_outlier_removal(scores, alpha=0.05):    
    
    lambda_diag = np.var(scores, axis=0, ddof=1)     # Calculating scores variance matrix 
    T2 = np.sum((scores**2) / lambda_diag, axis=1)     # T² values 
    
    # F-distribution threshold
    n, p = scores.shape
    F_crit = f.ppf(1 - alpha, p, n - p)
    threshold = (p * (n - 1) / (n - p)) * F_crit
    
    outliers = T2 > threshold #Thresholding
    scores = scores[~outliers]
    
    return scores, outliers, T2, threshold

class PCAExecutor(Executor):
    
    def __init__(self, train_task_name: str = "PCA_fit", data_similarity_task_name: str = "Data_similarity", standardization_task_name: str = "Data_standardization"):
        super().__init__()
        self._train_task_name = train_task_name
        self._data_similarity_task_name = data_similarity_task_name
        self._standardization_task_name = standardization_task_name
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        pass
    
    def execute(self, task_name:str, shareable: Shareable, fl_ctx: FLContext,abort_signal: Signal) -> Shareable:
        
        self.log_info(fl_ctx, f"Task {task_name} started.")
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            if task_name == self._standardization_task_name:
                return self.local_standardization(shareable, fl_ctx, abort_signal)
            elif task_name == self._train_task_name:
                return self.covariance(shareable,fl_ctx, abort_signal)
            elif task_name == self._data_similarity_task_name:
                return self.bounding_box_evaluation(shareable, fl_ctx,abort_signal)
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}. Exception: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)    
        
    def load_data(self, fl_ctx: FLContext):
        # 1. Load clinical dataset
        #self.data , _ , _ = ClinicalDataset(val_ratio = None, test_ratio=None )
        
        self.dataset = ClinicalDataset(val_ratio=None, test_ratio=None) 
        self.data = self.dataset.df_train  
        # 2. Keep only numeric columns (avoid dtype=object)
        #self.data = self.data.select_dtypes(include="number")
        # 2. Numpy array conversion
        self.data = np.array(self.data, dtype=float)
        self.log_info(fl_ctx, f"Site data shape: {self.data.shape}")
        
    def global_standardize_data(self, mean_std:Dict, data: np.ndarray, fl_ctx: FLContext , abort_signal: Signal) -> np.ndarray:
        
        global_mean = mean_std.get("global_mean")
        global_std = mean_std.get("global_std")
        #####################################################
        self.log_info(fl_ctx, f"Standardizing data with global mean: {global_mean} and global std: {global_std}")
        ######################################################
        data = (data - global_mean) /global_std 

        return data
    
    def local_standardization(self, shareable:Shareable, fl_ctx: FLContext , abort_signal: Signal) -> Shareable:
        
        ###################################
        self.log_info(fl_ctx, f" Calculating local count, mean and std ...")
        ###################################
        n_mean_std =  from_shareable(shareable).data # empty shareable from server 
        # 1. Load data
        self.load_data(fl_ctx)        
        # 2. Local stats
        local_count = self.data.shape[0]
        local_mean = np.mean(self.data, axis=0)
        local_std = np.std(self.data, axis=0)
        # 3. DXO to send to server
        n_mean_std["local_count"] = local_count
        n_mean_std["local_mean"] = local_mean
        n_mean_std["local_std"] = local_std
        ##############################################
        self.log_info(fl_ctx, f" Resulting local count {local_count}, mean {local_mean} and std {local_std}")
        ##############################################
        dxo = DXO(data_kind="WEIGHTS", data=n_mean_std)
        shareable = dxo.to_shareable()
        shareable.set_header("status", TaskCompletionStatus.OK)
        
        return shareable

    def covariance(self, shareable:Shareable, fl_ctx: FLContext , abort_signal: Signal) -> Shareable:
        
        self.log_info(fl_ctx, f"Evaluating local non normalized covariance...")
        n_mean_std = from_shareable(shareable).data.get("n_mean_std").get("numpy_key")
                
        # 1. Load data and standardize 
        self.load_data(fl_ctx)
        self.global_standardize_data(n_mean_std, self.data, fl_ctx, abort_signal)
        # 2. Local stats
        local_count = self.data.shape[0]
        local_mean = np.mean(self.data, axis=0)
        cov_matrix = np.dot((self.data - local_mean).T, (self.data - local_mean)) # non normalized covariance
        # 3. DXO to send to server
        nn_cov = {"numpy_key":{
            "local_count": local_count,
            "local_mean": local_mean,
            "cov_matrix": cov_matrix
        }}
        
        data = {
                    "global_n_mean_std": from_shareable(shareable).data.get("n_mean_std"),
                    "covariance": nn_cov
                }
        dxo = DXO(data_kind="WEIGHTS", data=data)
        shareable = dxo.to_shareable()
        shareable.set_header("status", TaskCompletionStatus.OK)
        return shareable
    
    def bounding_box_evaluation(self,PC_shareable:Shareable, fl_ctx: FLContext, abort_signal: Signal):
        
        self.log_info(fl_ctx, f"Receiving aggregated loadings...")
        # Open DXO from server
        dxo = from_shareable(PC_shareable).data.get("numpy_key")
        global_n_mean_std = dxo["global_n_mean_std"]
        loadings = dxo["loadings"]
                
        # 1. Load data and standardize 
        self.load_data(fl_ctx)
        self.global_standardize_data(global_n_mean_std, self.data, fl_ctx, abort_signal)
        # 2. Obtain X_scores and filter outliers 
        X_scores = np.dot(self.data, loadings)
        X_scores_filtered, _ , _ , _ = hotelling_t2_outlier_removal(X_scores, alpha=0.05)
        
        #3. Define Bounding Box and volume
        x_max = X_scores_filtered.max(axis = 0)
        x_min = X_scores_filtered.min(axis = 0)
        vol =  np.prod(abs(x_max-x_min))
        #########################################
        self.log_info(fl_ctx, f" Results: x_min {x_min}, x_max {x_max} and volume {vol}")
        #########################################
        # 4. DXO to send to server
        result = {
            "x_max": x_max,
            "x_min": x_min,
            "vol": vol
        }
        dxo = DXO(data_kind="PC_VOLUME", data=result)
        shareable = dxo.to_shareable()
        shareable.set_header("status", TaskCompletionStatus.OK)
        return shareable
    
    
