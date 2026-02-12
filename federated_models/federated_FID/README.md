## Frechét Inception Distance distributed across sites

### Description
This experiment computes the Fréchet Inception Distance (FID) in a distributed setting across multiple sites using NVFlare.
Each site computes local feature statistics, which are then aggregated to estimate inter-site domain differences without sharing raw data.

### Output
The experiment outputs the computed FID scores between sites.
Results are saved to disk as numpy array and can be used to analyze data heterogeneity and domain shifts across participating sites.

To use in the Ippocrate NVFlare environment, install the requirements with 
```shell
pip install -r ./federated_FID/requirements.txt
```
then run the experiment. 

