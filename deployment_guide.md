# Deploying the Agent with Docker

This guide provides step-by-step instructions on how to containerize the agent using Docker, test it locally, and outlines the general process for deploying it to a cloud hosting service.

## Prerequisites

1.  **Docker:** Ensure Docker is installed and running on your local machine. You can download it from the [official Docker website](https://www.docker.com/get-started).
2.  **Container Registry Account (Optional but Recommended):** If you plan to deploy to the cloud, you'll need an account with a container registry like [Docker Hub](https://hub.docker.com/), [Google Container Registry (GCR)](https://cloud.google.com/container-registry), [Amazon Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/), or [Azure Container Registry (ACR)](https://azure.microsoft.com/en-us/services/container-registry/).

## Step 1: Create a `Dockerfile`

A `Dockerfile` contains instructions for building a Docker image. Create a file named `Dockerfile` (no extension) in the root directory of your project.

```Dockerfile
# 1. Choose a base image
# Use an official Python runtime as a parent image. Choose a specific version.
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy dependency files and install dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Expose the port the app runs on (if it's a web service)
# Replace 8000 with the actual port your agent listens on, if applicable.
# If your agent is not a long-running service or doesn't expose a port, you might omit this.
# EXPOSE 8000

# 6. Define the command to run the application
# Replace 'python', 'your_agent_script.py' with the actual command to start your agent.
# If your agent needs environment variables, ensure they are provided at runtime (see Step 4).
CMD ["python", "your_agent_script.py"]
```

**Notes:**

- Replace `python:3.10-slim` with the specific Python version your project uses.
- Ensure you have a `requirements.txt` file listing all Python dependencies. You can generate one using `pip freeze > requirements.txt`.
- Modify the `EXPOSE` command if your agent listens on a different port or doesn't listen on any port.
- Update the `CMD` instruction to the correct command for starting your agent's main script.

## Step 2: Build the Docker Image

Open your terminal, navigate to the project's root directory (where the `Dockerfile` is located), and run the build command:

```bash
docker build -t your-agent-image-name:latest .
```

- Replace `your-agent-image-name` with a suitable name for your image (e.g., `paperboy-agent`).
- `:latest` is a tag, often used for the latest version. You can use specific version tags (e.g., `:v1.0`).
- The `.` at the end specifies the build context (the current directory).

This command will execute the instructions in your `Dockerfile`, download the base image, install dependencies, and create your application image.

## Step 3: Run the Docker Container Locally

Before pushing to the cloud, test the image locally:

```bash
docker run --rm -it \
  # -p 8000:8000 \  # Uncomment and adjust if you exposed a port
  # -e YOUR_ENV_VAR=your_value \ # Add environment variables if needed
  your-agent-image-name:latest
```

- `docker run`: Command to run a container from an image.
- `--rm`: Automatically removes the container when it exits.
- `-it`: Runs the container in interactive mode with a pseudo-TTY (useful for seeing logs/output). Use `-d` for detached mode if it's a long-running service.
- `-p 8000:8000`: (Optional) Maps port 8000 on your host machine to port 8000 in the container. Adjust the ports if necessary based on the `EXPOSE` instruction in your `Dockerfile`.
- `-e YOUR_ENV_VAR=your_value`: (Optional) Use this flag to pass environment variables required by your application. It's highly recommended to manage secrets securely (e.g., using `.env` files mounted as volumes or cloud provider secret management) rather than passing them directly here in production.
- `your-agent-image-name:latest`: The name and tag of the image you built.

Verify that the agent starts correctly and functions as expected within the container. Check the logs for any errors.

## Step 4: Push the Image to a Container Registry (Optional)

To deploy to a cloud service, you typically need to push your image to a registry.

1.  **Log in to your registry:**

    - **Docker Hub:** `docker login`
    - **GCR:** `gcloud auth configure-docker` (requires gcloud CLI)
    - **ECR:** `aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-aws-account-id.dkr.ecr.your-region.amazonaws.com` (requires AWS CLI)
    - **ACR:** `az acr login --name your-registry-name` (requires Azure CLI)

2.  **Tag your image:** You need to tag your local image with the registry path.

    - **Docker Hub:** `docker tag your-agent-image-name:latest your-dockerhub-username/your-agent-image-name:latest`
    - **GCR:** `docker tag your-agent-image-name:latest gcr.io/your-gcp-project-id/your-agent-image-name:latest`
    - **ECR:** `docker tag your-agent-image-name:latest your-aws-account-id.dkr.ecr.your-region.amazonaws.com/your-agent-image-name:latest`
    - **ACR:** `docker tag your-agent-image-name:latest your-registry-name.azurecr.io/your-agent-image-name:latest`
    - Replace placeholders with your actual registry details, usernames, project IDs, etc.

3.  **Push the image:**
    ```bash
    docker push your-registry-path/your-agent-image-name:latest
    ```
    - Use the fully qualified tag you created in the previous step.

## Step 5: Deploy to a Cloud Hosting Service

The exact steps depend heavily on your chosen cloud provider and service. Here are general pointers for common services:

- **Google Cloud Run:**
  - A fully managed serverless platform.
  - Deploy directly from an image in GCR.
  - Configure CPU, memory, scaling, environment variables, and secrets via the Cloud Console or `gcloud` CLI.
  - Ideal for stateless applications that can start quickly.
- **AWS Elastic Container Service (ECS):**
  - An orchestrator for Docker containers.
  - Requires defining Task Definitions (describing your container, resources, image from ECR) and Services (managing running instances of tasks).
  - Can run on EC2 instances (you manage) or Fargate (serverless).
- **Azure Container Instances (ACI):**
  - Run single Docker containers without orchestration.
  - Simple deployment from ACR or Docker Hub.
- **Azure App Service:**
  - Platform-as-a-Service (PaaS) that supports containers.
  - Deploy from ACR, Docker Hub, or other registries.
  - Provides scaling, load balancing, and deployment slots.
- **Kubernetes (GKE, EKS, AKS):**
  - A powerful container orchestration platform.
  - Steeper learning curve but offers maximum flexibility and scalability.
  - Requires defining Deployment objects (to manage container replicas) and Service objects (to expose your application).

**General Cloud Deployment Considerations:**

- **Environment Variables & Secrets:** Use the cloud provider's recommended mechanism for managing sensitive information (e.g., Secret Manager, Parameter Store, Key Vault) instead of hardcoding or putting them directly in the container image or environment variables.
- **Logging & Monitoring:** Configure your application to output logs to standard output/error and integrate with the cloud provider's logging service (e.g., Cloud Logging, CloudWatch, Azure Monitor).
- **Networking:** Configure firewall rules, load balancers, and VPCs/VNets as needed.
- **Databases & Storage:** If your agent needs persistent storage or interacts with a database, provision and configure these resources separately in the cloud and provide connection details securely to your container.
- **CI/CD:** Set up a Continuous Integration/Continuous Deployment pipeline (e.g., using GitHub Actions, GitLab CI, Cloud Build, Jenkins, Azure DevOps) to automate building, testing, and deploying your container whenever you push changes to your code repository.

This guide provides a solid foundation. Remember to consult the specific documentation of your chosen container registry and cloud hosting service for detailed instructions tailored to their platform.
