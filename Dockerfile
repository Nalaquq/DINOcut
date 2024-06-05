# Use the official Ubuntu 22.04 LTS (jammy) image as the base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and other dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages using setup.py
RUN python3 setup.py install

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["python3", "dinocut.py"]
