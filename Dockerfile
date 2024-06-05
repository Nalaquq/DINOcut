# Use the official Ubuntu 22.04 LTS (jammy) image as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common

# Add deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and necessary tools
RUN apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages using setup.py
RUN python3.10 setup.py install

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["python3.10", "dinocut.py"]
