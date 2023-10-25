### Surface Refinement Docker

#### Overview

This documentation offers a detailed guide on setting up a remote Docker development environment optimized for Surface Refinement using CUDA. By executing a simple bash script, developers can efficiently establish their environment and begin their development activities.

#### Supported Platforms
- **WSL2**
- **Linux**

#### Supported CPU Architectures
- **AMD64**

#### Prerequisites

Before proceeding, ensure the installation of the following software:
- Docker
- Docker-compose

#### Setup and Execution

1. Navigate to the appropriate project directory.
2. Run the following command:
    ```bash
    bash docker/start_docker_compose.sh
    ```

#### Configuration Guidelines

- The `.env` file contains a range of configurable parameters, such as `ROOT_PASSWORD`, `PUBLIC_KEY`, `CLION_VERSION`, `GIT_USER_NAME`, `GIT_USER_EMAIL`, `SSH_PORT`, and `XRDP_PORT`.

- Some packages, like `zip` and `glances`, are specified in the Dockerfile. While they provide developmental benefits, they may not be essential for every scenario. Developers can choose to exclude them if unnecessary.

##### .env Configuration Details

- `ROOT_PASSWORD`: Sets the root password for the Docker container.
- `PUBLIC_KEY`: Indicates the public key for the Docker container.
- `GIT_USER_NAME`: Represents your Git username.
- `GIT_USER_EMAIL`: Represents your Git email address.
- `SSH_PORT`: Assigns the SSH port for the Docker container.
- `XRDP_PORT`: Sets the XRDP port for the Docker container.

#### Development Usage

- Developers can access the Docker container via XRDP and continue their work in CLion.

- Alternatively, if one has a preferred IDE, remote development through SSH is a viable choice.
