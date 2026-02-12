import numpy as np
from itertools import combinations_with_replacement

from nvflare.apis.dxo import DXO
from nvflare.apis.dxo import DataKind
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.app_common.abstract.aggregator import Aggregator


def find_overlap(intervals):
        """ Find overlap across bounding boxes """
        try:
            x_min_max = np.max([interval[0] for interval in intervals], axis=0) # max of mins
            x_max_min = np.min([interval[1] for interval in intervals], axis=0) # min of maxs
            V1_U_V2 = np.sum([interval[2] for interval in intervals])
            mask = x_min_max <= x_max_min
            if not np.all(mask):
                return np.zeros_like(x_min_max), V1_U_V2 #return zero vector if no overlap

            overlap = x_max_min - x_min_max
            
            return overlap , V1_U_V2
        except Exception as e:
            print(f"Error in finding overlap: {e}")
            return None, None
        
def find_all_overlaps(datasets):
        """ Compute similarities between dataset pairs """
        try:
            N = len(datasets)
            similarity_matrix = np.zeros((N, N))
            overlaps = {}
            combinations = list(combinations_with_replacement(enumerate(datasets), 2))
            
            for el in combinations: 
                i, ds1 = el[0]
                j, ds2 = el[1]
                overlap, V1_U_V2 = find_overlap([ds1, ds2])
                overlap = np.prod(abs(overlap)) # Compute overlapping volume 
                norm_overlap = 2*overlap/V1_U_V2
                similarity_matrix[i, j] = norm_overlap
                similarity_matrix[j, i] = norm_overlap
                overlaps[f"Dataset {i} - Dataset {j}"] = (overlap, norm_overlap)
            return overlaps, similarity_matrix
        except Exception as e:
            print(f"Error in finding overlaps: {e}")
            return None, None
    
def pca_numpy(corr_matrix, threshold=0.7):
    """
    Principal components analysis from correlation matrix in NumPy.
    """
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)  # Correlation matrix eigvalues and eigvectors
    idx = np.argsort(eigvals)[::-1]  # Sort eigvalues in descending order
    principal_components = eigvecs[:, idx]  
    
    explained_variance = eigvals[idx]/ np.sum(eigvals)  # explained variance 
    cumulative_variance = np.cumsum(explained_variance)  # cumulative explained variance 
    
    # Find number of components to explain at least the threshold variance
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    return principal_components[:, :n_components], explained_variance[:n_components], cumulative_variance[n_components-1]


class PCAAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        
        self.task = ""
        self.results = {} 
        self.variance_threshold = 0.7 # threshold for PCA explained variance     

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        peer_ctx = fl_ctx.get_peer_context()
        client_name = peer_ctx.get_identity_name()
                
        if self.task not in self.results:
            self.results[self.task] = {}
            
        if client_name not in self.results[self.task]:
            self.log_info(fl_ctx, f"--------------- TASK {self.task} from {client_name} ----------------")
            self.results[self.task][client_name] = shareable
            return True
        else:
            self.log_warning(fl_ctx, f"Data for {client_name} already exists.")
        
    def global_standardization(self, fl_ctx: FLContext):
        
        self.task = "Data_standardization"
        try:
            sites_data = []
            for client_name, reply in self.results[self.task].items():
                
                if reply.get_header('status') == TaskCompletionStatus.OK:
                    client_result = reply.get("DXO").get("data")
                    sites_data.append((
                        client_result.get("local_count"),
                        client_result.get("local_mean"),
                        client_result.get("local_std")
                    ))                
                else:
                    self.log_warning(fl_ctx, f"{client_name} ha fallito il task")
                    
            N = sum(n_i for n_i, _, _ in sites_data) #global count
            mu_global = sum(n_i * mu_i for n_i, mu_i, _ in sites_data) / N # global mean
            sigma_global = np.sqrt(sum(n_i * (sigma_i**2 + (mu_i - mu_global)**2) for n_i, mu_i, sigma_i in sites_data) / N ) # global std
            
            data = {"numpy_key":{
                "global_count": N,
                "global_mean": mu_global,
                "global_std": sigma_global}}
            
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
            return dxo.to_shareable()
                
        except Exception as e:
            self.log_error(fl_ctx, f"Error in aggregation: {e}")
            return None
                    
    def aggregate(self, fl_ctx: FLContext):
        """
        Aggregate non-normalized covariance results from all clients.
        """
        
        self.task="PCA_fit"
        try:
            sites_stats = []
            for client_name, reply in self.results[self.task].items():

                if reply.get_header('status') == TaskCompletionStatus.OK:
                
                    client_result = reply.get("DXO").get("data").get("covariance").get("numpy_key")
                    
                    sites_stats.append((
                        client_result.get("local_count"),
                        client_result.get("local_mean"),
                        client_result.get("cov_matrix")

                    ))            
                else:
                    self.log_warning(fl_ctx, f"{client_name} : task failed.")
                    
            # Global metrics 
            N = sum(n_i for n_i, _, _ in sites_stats) #Global count
            mu_global = sum(n_i * mu_i for n_i, mu_i, _ in sites_stats) / N # Global mean
            S_global = sum(S_i for _, _, S_i in sites_stats) + sum(
                n_i * np.outer(mu_i - mu_global, mu_i - mu_global)
                for n_i, mu_i, _ in sites_stats
            ) 
            covariance_matrix = S_global / (N - 1)  # Normalize #Global covariance matrix
            std_devs = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix / (std_devs[:, None] * std_devs[None, :]) #Global correlation matrix
            
            # PCA calculation
            loadings, e_v,c_v  = pca_numpy(correlation_matrix, threshold=self.variance_threshold)
            self.log_info(fl_ctx, f"Total explained variance is: {c_v}, number of components {loadings.shape[1]}, explained variance {e_v}." )
            self.log_info(fl_ctx, f"End aggregation.")
            
            global_n_mean_std =  reply.get("DXO").get("data").get("global_n_mean_std").get("numpy_key")
            data = {"numpy_key":{"global_n_mean_std": global_n_mean_std,
            "loadings": loadings}}
            
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
            return dxo.to_shareable()
        
        except Exception as e:
            self.log_error(fl_ctx, f"Error in aggregation: {e}")
            return None
    
    def compare(self, fl_ctx: FLContext):
        
        self.task="Data_similarity"
        
        try:
            sites_results = []
            client_names = []

            for client_name, reply in self.results[self.task].items():
                if reply.get_header('status') == TaskCompletionStatus.OK:
                    
                    client_names.append(client_name)
                    client_result = reply.get("DXO").get("data")
                    
                    x_min = client_result.get("x_min")
                    x_max = client_result.get("x_max")
                    vol = client_result.get("vol")
                    
                    sites_results.append((
                        x_min,
                        x_max,
                        vol
                    ))
                else:
                    self.log_warning(fl_ctx, f"{client_name}: task failed.")
                    
            
            self.log_info(fl_ctx, f"--------------- TASK {self.task} ----------------")
            
            _ , similarity_matrix = find_all_overlaps(sites_results)

            similarity = {"numpy_key": {
            "similarity_matrix": similarity_matrix,
            "Client order: ": client_names
            }}
        
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=similarity)
            return dxo.to_shareable()
            
        except Exception as e:
            self.log_error(fl_ctx, f"Error in comparison: {e}")
            return "Error in comparison"
    