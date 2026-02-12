# Ippocrate

IPPOCRATE — Infrastructure for Privacy Preserving Orchestration of Clinical ReseArch for Trustworthy & Explainable AI — is a research-driven platform designed to advance federated learning (FL) within the healthcare domain. Federated learning is a transformative approach in artificial intelligence that enables multiple participants to collaboratively train models without transferring or exposing their local datasets. This is particularly vital in clinical research, where maintaining patient privacy and ensuring data security are both ethical imperatives and regulatory requirements.

IPPOCRATE addresses these needs by providing a secure and scalable infrastructure tailored to the unique challenges of medical data. The platform orchestrates distributed AI training across multiple healthcare institutions while ensuring that sensitive patient data remains confined to its source. Its architecture supports end-to-end secure communication, robust orchestration of model training, and flexible configuration of learning workflows using a range of advanced algorithms.

The platform is also equipped with modules that facilitate interoperability across institutional systems. These components support integration with diverse data formats and schemas, helping harmonize heterogeneous medical data to ensure smooth collaboration. IPPOCRATE further empowers users by offering pre-trained models that can be customized and deployed for inference within the platform itself.

Altogether, IPPOCRATE creates a comprehensive ecosystem for privacy-preserving, cross-site AI development. By enabling medical institutions to collaboratively build explainable and trustworthy models, it advances clinical research while upholding the highest standards of data confidentiality and regulatory compliance.

## Project Structure

The IPPOCRATE project is organized into four main directories, each responsible for a specific part of the federated learning and clinical data processing workflow.

The **federated_models** directory contains all the code necessary for the federation server to operate. This includes the logic and interfaces required to distribute and manage training jobs across participants, as well as coordinate the lifecycle of distributed AI models. It represents the core engine of the collaborative learning process.

The **nvflare** directory provides the setup instructions and resources needed to configure both the server and client environments that will host the federation. This section includes setup scripts, configuration files, and all essential components to deploy and maintain the federated network using the NVIDIA FLARE platform.

The **omop** directory includes a guide and necessary resources to install and configure the OMOP database. This database serves as the standardized repository for patient data within each participating institution, enabling semantic and structural consistency across sites and ensuring interoperability.

The **processing** directory contains all tools and scripts required for data preprocessing. These resources are designed to transform raw datasets into the format required by the OMOP schema, allowing proper data ingestion and alignment with the local databases of each client institution.
# Ippocrate-bracco
