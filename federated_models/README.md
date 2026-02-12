# Federated Models

This folder contains jobs for:
- **genomic_regressor**: Federated logistic regression for VCF2Matrix-like data
- **covid**: Federated classifier to distinguish between "covid" and "normal" chest X-rays. The dataset is stored in a CSV file which contains pointers to the images.
- **covid_omop**: Same as above, but the dataset is read directly from the OMOP database. The setup for this job is described in `~/omop/IPPOCRATE_omop_db/README.md`
- **federated_PCA**: federated principal component analysis for inter-sites tabular data.
- **federated_FID**: inter-sites Frechét Inception Distance.
- **federated_UNET**: 2D/3D UNet architecture and training for medical image segmentation.
  
To run the jobs, move the corresponding folder into the `/transfer` directory of the server kit and submit the job from the NVflare CLI.

## Genomic Regression

Generate the dataset using the Python script `~/federated_models/genomic_regressor/data/generate_data.py`. Once generated, move the training and validation datasets of each client to the folder of the corresponding workstation that binds to the "/home/data" directory in the NVFLARE container. Name each training dataset as "training_data_client.csv" and each validation dataset as "validation_data_client.csv" (refer to lines 36-37 of `~/federated_models/genomic_regressor/app/custom/genomic_datamanager.py` to customize this part).

## COVID

Download the chest X-rays from the [public dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) to the folder corresponding to the container path specified in line 112 of `~/federated_models/covid/app/custom/chest_xray_datamanager.py`. The pointers to the images, labels, training-validation split, and client split are defined in a CSV file which must be placed according to line 113 of `~/federated_models/covid/app/custom/chest_xray_datamanager.py`. You can find an example of this file in `~/federated_models/covid/data`.

## COVID OMOP

This setup requires having the compose defined in `~/omop/IPPOCRATE_omop_db/README.md` up and running, and the OMOP database populated with patient images.




