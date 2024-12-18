# torchrun --standalone --nnodes=1 --nproc_per_node=2 distributed-training-2.py
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install PyTorch and dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install matplotlib boto3

# Install procps for debugging (optional)
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Copy your training script
COPY ./src/distributed-training.py /app/

# Copy the data in the container
COPY ./MNIST /app/MNIST

# Set environment variables
ENV PYTHONUNBUFFERED=1
#ENV MASTER_PORT=29500

# Only set ENTRYPOINT, let Kubernetes control the arguments
ENTRYPOINT ["torchrun"]
CMD ["--nnodes=1", "--nproc_per_node=2", "distributed-training.py"]

