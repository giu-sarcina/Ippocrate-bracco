# NVFlare

From a technical standpoint, IPPOCRATE is built on NVIDIA FLARE (Federated Learning Application Runtime Environment), a domain-agnostic software development kit (SDK) that allows researchers to adapt machine learning and deep learning workflows to a federated paradigm. FLARE provides built-in support for secure communication, privacy preservation, and scalable orchestration of training tasks across distributed environments.

A typical federated learning experiment involves a central server that coordinates a group of distributed clients. This process unfolds through three main phases. In the Provision phase, the project environment is initialized, and unique identities are assigned to each client. During the Start phase, federated training is launched, with the server and clients engaging in synchronized communication and computation cycles. The Operate phase covers the ongoing execution and monitoring of training across the entire federated network, ensuring consistency, fault tolerance, and traceability.

NVIDIA FLARE natively supports all these phases and offers a rich set of federated learning workflows. These include server-side controlled strategies such as scatter-and-gather, cyclic weight transfer, federated evaluation, and cross-site model evaluation. On the client-side, it enables autonomous control through approaches like cyclic weight transfer, swarm learning, and independent evaluation across sites. In addition, FLARE supports split learning, where model components are partitioned and trained across different nodes to reduce exposure of sensitive information.

The platform also integrates a variety of learning algorithms, including FedAvg, FedOpt, FedProx, Scaffold, Ditto, FedSM, and FedAutoRL. To enhance data privacy, FLARE supports techniques such as homomorphic encryption and differential privacy, which allow secure computation and minimize data leakage risks.

## Setup

To set up the federated learning environment in IPPOCRATE, two Docker images are required:

1.  `nvflare/nvflare` for the FLARE dashboard and admin tools.

    ```bash
    docker pull nvflare/nvflare
    ```

2.  A custom image based on NVIDIA PyTorch 24.05, used by both the federation server and clients. This custom image is built from the official NVIDIA PyTorch container `NVIDIA PyTorch 24.05` (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html), which includes CUDA 12.4 support. If you're using a different CUDA version, you should select a compatible base image from the NVIDIA PyTorch release catalog. On top of this base, essential network tools and Python libraries required for data processing and federated learning are added. The image also installs nvflare (version 2.5.2) and any additional dependencies defined in a `requirements.txt` file. To build the image, use:

    ```bash
    docker build -t nvflare-cuda124 .
    ```

## Dashboard

First, download the nvflare Docker image and run a container for the dashboard, binding it to port `8443` on the server (e.g., `iitfedlearn.iit.it`):

```bash
docker run -td --name nvidia-dashboard -p 8443:8443 nvflare/nvflare
docker exec -it nvidia-dashboard /bin/bash
```

Inside the container, launch the FLARE dashboard using the following command:

```bash
nvflare dashboard --local
```

You will be prompted to enter an admin email whereas a password will then be automatically generated. Make sure to save this password, as it is required to access the dashboard. Once launched, you can access the dashboard through a web browser by visiting `https://iitfedlearn.iit.it:8443`. Log in using the admin email and the generated password.

Once authenticated as the project admin (typically the server-side role), which is also responsible for approving the creation of user accounts and associated client sites, edit the project description and set the project visibility to public to allow user sign-ups. Hence, create the necessary user accounts for the client roles and for the admin organization (admin org). The admin org is responsible for managing the client sites that will later participate in the federation.

> [!IMPORTANT]  
> When logging into the dashboard, always use the email address, not the username.

Before freezing the project, configure the DNS name (e.g., `iitfedlearn.iit.it`) to allow proper package downloads. Once the DNS is set, freeze the project to generate the downloadable starter kits for each role: Clients, Client Sites, Admin Org and Server. Do not forget to save all the passwords provided during the kit download process, as they will be needed later when unzipping the packages.

Unzip each downloaded kit using the corresponding password. Extract all kits into a single directory called kits and transfer the entire kits folder to the machine where the federation clients or server will be deployed. Each machine should receive only the relevant kit (client, client-site, admin org, or server), depending on its role in the federation. For instance, the commands to launch for the server are:

```bash
mkdir kits
cd kits
unzip <server-name>.zip
scp -r <server-name> <remote-user>@<remote-host>:/home/<remote-user>
```

## Server

To launch the FLARE federation server using the custom Docker image (nvflare-cuda124), execute the following commands:

```bash
docker run -td --rm \
  --name nvflare-server \
  -p 8002:8002 -p 8003:8003 \
  -w /home/<server-name> \
  --mount type=bind,src=/home/<remote-user>,dst=/home \
  nvflare-cuda124
docker exec -it nvflare-server /bin/bash
```

These commands start a container named nvflare-server, bind the necessary ports (`8002` and `8003`), and mount the host directory `/home/<remote-user>` into the container's `/home` directory. The working directory inside the container is set to `/home/<server-name>`.
Starting the Server

Once inside the container shell, navigate to the project folder `<server-name>` and execute the `start.sh` script to launch the server:

```bash
cd <server-name>
./start.sh
```

If you encounter issues related to the Python environment (e.g., incorrect interpreter or missing dependencies), you may need to modify the `sub_start.sh` script. Specifically, update the `PYTHONPATH` variable with the correct path to the Python executable. To identify the correct Python path inside the container, use:

```bash
which python
```

Then edit `sub_start.sh` to set `PYTHONPATH` accordingly. This ensures that the correct Python environment is used when launching the federation server.

## Client

To launch the FLARE federation server using the custom Docker image (nvflare-cuda124), execute the following commands:

```bash
docker run --gpus all  \
    -td --rm --name nvflare-client \
    -w /home/<client-name>  \
    --mount type=bind,src=/home/<remote-user>,dst=/home \
    nvflare-cuda124
docker exec -it nvflare-client /bin/bash
```

Once inside the container shell, navigate to the project folder `<client-name>` and execute the `start.sh` script to launch the client:

```bash
cd <client-name>
./start.sh
```

If you encounter issues related to the Python environment (e.g., incorrect interpreter or missing dependencies), you may need to modify the `sub_start.sh` script. Specifically, update the `PYTHONPATH` variable with the correct path to the Python executable. To identify the correct Python path inside the container, use:

```bash
which python
```

You may also need to adjust the GPU configuration used by the system. Navigate to the project directory and locate the file `local/resources.json.default`. Open this file and edit the configuration by setting the `num_of_gpus` property to the number of GPUs available on the host system. Make sure to use an **integer**, not a string.

## NVFlare CLI

At this point, you can interact from inside the client or the server container with the FLARE system using the Admin CLI, provided in the `startup` folder of the Admin Org's starter kit. To launch the CLI, run the `fl_admin.sh` script. You will be prompted to enter the email address associated with the Admin Org user. This is used for authentication when connecting to the FLARE server.

## Job Submission

Before submitting a job in the FLARE ecosystem, it is essential to configure the runtime environment within the `local` directory of the server kit.

To enable detailed logging at the `DEBUG` level, start by copying the default logging configuration file. Duplicate `log.config.default` and rename it to `logging.conf`. Then, open this file and modify the root logger's level to `DEBUG`. This ensures that all system messages, including those useful for troubleshooting, are captured during execution.

Next, create the `resources.json` file in the same `local` directory. This file is crucial, as it instructs the server which job manager and supporting components to use. It also defines runtime settings such as the job scheduler configuration, client behavior, and snapshot persistence.
Clients must also have their own `resources.json` file, tailored to the hardware and resources available on their side. Without this file, the FLARE client cannot correctly allocate or request computing resources, which can lead to errors during training.

To submit a job, first copy its corresponding folder from the federated_models folder in this repository into the `transfer` directory of the Admin Console kit. Each job must be placed inside a folder whose name matches the job’s identifier. Within this folder, there must be a subdirectory named `app`, which should be organized into two directories: `config`, containing the `config_fed_client.json` and `config_fed_server.json` files, and `custom`, which stores custom Python components referenced in the configurations. The job directory must also include a `meta.json` file. This file defines metadata for the job, including its name and deployment parameters. Before proceeding, ensure that client names listed in `meta.json` (e.g., `client1_site`) exactly match those used in the federation’s configuration.

Once the job directory is correctly set up, it can be submitted using the Admin CLI with the command `submit_job <job_name>`. Upon submission, the job code is distributed to participating clients and stored in a folder named after the `job_id`. This folder will also be used to store log files generated during the execution of the job by the clients.

If a job submission results in a timeout, the recommended approach is to restart both the server and the clients before attempting to resubmit the job. This helps resolve potential synchronization or resource allocation issues that may have occurred during the initial submission.
