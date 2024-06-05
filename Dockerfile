FROM ubuntu:22.04
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3
RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y python3 \
  && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Nalaquq/DINOcut.git
#WORKDIR /DINOcut
CMD ["python3", "setup.py"]
