## Federated PCA across sites

### Description
This experiment computes Principal Component Analysis (PCA) in the federated setting as described in []. 
Each site computes local feature statistics, which are then aggregated to estimate inter-site domain differences without sharing raw data.

### Output
The experiment outputs the PCA resulting distance matrix beatween sites. Results are saved to disk as numpy array and can be used to analyze data heterogeneity and domain shifts across participating sites.

To use in the Ippocrate NVFlare environment, install the requirements with 
```shell
pip install -r ./federated_PCA/requirements.txt
```
then run the experiment. 
