import numpy as np
from scipy import linalg
from itertools import combinations

from nvflare.apis.dxo import DXO
from nvflare.apis.dxo import DataKind
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.app_common.abstract.aggregator import Aggregator

# Function to calculate FID
def calculate_fid(res1 , res2):
    mu1, sigma1 = res1["mean"],res1["cov"] 
    mu2, sigma2 = res2["mean"],res2["cov"]
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # Calculate sqrt of product between covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    
def calculate_fid_matrix(sites_results):
    """
    Calculate the FID matrix for a list of sites results.
    
    Args:
        sites_results (list): List of tuples containing mu and sigma for each site and series.
        
    Returns:
        dict of np.ndarray: FID matrices, one for series 
    """
    series_similarities = {}
    clients_names = list(sites_results.keys())

    # lista serie (assumo stessa struttura per tutti)
    first_client = clients_names[0]
    series_names = sorted(sites_results[first_client].keys())

    for name in series_names:
        # prendo i client che hanno quella serie
        series_clients = [c for c in clients_names if name in sites_results[c]]

        n = len(series_clients)
        distance_matrix = np.zeros((n, n))

        # combinazioni di indici
        for i, j in combinations(range(n), 2):
            c1, c2 = series_clients[i], series_clients[j]

            res1 = sites_results[c1][name]
            res2 = sites_results[c2][name]
            dist = calculate_fid(res1, res2)

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

        # normalizzazione
        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            similarity = 1 - (distance_matrix / max_dist)
        else:
            similarity = np.ones_like(distance_matrix)

        similarity = np.round(similarity, 4)
        series_similarities[name] = similarity

    return series_similarities, clients_names


class FIDAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        
        self.task = ""
        self.results = {}     

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
    
    def aggregate(self,fl_ctx):
        
        self.task="FID_to_similarity"
        try:
            sites_results = {}

            for client_name, reply in self.results[self.task].items():
                if reply.get_header('status') == TaskCompletionStatus.OK:
                    series = reply.get("DXO").get("data")
                    #series = client_result.get("numpy_key")
                    sites_results[client_name] = series 
                else:
                    self.log_warning(fl_ctx, f"{client_name} failed the task.")
                    
            sites_results = dict(sorted(sites_results.items())) # sorting 
            self.log_info(fl_ctx, f"--------------- TASK {self.task} ----------------")
            
            series_similarities, clients_names = calculate_fid_matrix(sites_results)
            
            similarity = {"numpy_key": {
            "FID_series_similarity_matrices": series_similarities,
            "Ordered clients list: ": clients_names
            }}
        
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=similarity)
            return dxo.to_shareable()
            
        except Exception as e:
            self.log_error(fl_ctx, f"Error in comparison: {e}")
            return "Error in comparison"
    