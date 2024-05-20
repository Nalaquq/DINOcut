
<div align="center">
    <img src="./assets/DINOcut_thumbnail.png" width="40%">
</div>


# 🦖 DinoCut ✂️

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=Cf0wft5CKT4) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xbq9rEOtyQh8QUQU-__E-Ub3Wy3X1NoV)[![Static Badge](https://img.shields.io/badge/GroundingDINO-arXiv-blue)](https://arxiv.org/abs/2303.05499) [![Static Badge](https://img.shields.io/badge/Segment_Anything-arXiv-blue)](https://arxiv.org/abs/2304.02643) [![Static Badge](https://img.shields.io/badge/Cut_Paste_Learn-arXiv-blue)](https://arxiv.org/abs/1708.01642) [![Static Badge](https://img.shields.io/badge/Grounded_SAM-arXiv-blue)](https://arxiv.org/abs/2401.14159)



[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

#  Usage
insert video of it working...recorded shell script..

# 🧠 Conceptual Design 📚 
The goal of this project is simple: to combine sommething old with soemthing new. So we've created an image processing pipeline for object detection using Grounding DINO; SAM; and a cut, paste learn (BGcut) approach. The result is a semi-supervised image processing pipeline that allows users to generate large, synthetic datasets for object detection without the hassle of manually labeling bounding boxes or creating segmentation masks. 

<div align="center">
    <img src="assets/DINOcut fig 1.png" width="100%">
</div>
 
# 🔧 Install 

**Installation:**

1.Clone DinoCut from GitHub.

```bash
git clone https://github.com/Nalaquq/cmm.git
```

2. Install the required dependencies.

```bash
pip install -e .
```

# 😳 Troubleshooting 
**Virtual Environments:**

We strongly encourage the use of a [python virtual environment](https://docs.python.org/3/library/venv.html) to manage packages and paths. To create a virtual environment use the following command: 

```bash
python3 -m venv venv
```

To activate your virtual environment use the following command at the beginning of each session: 

```bash
python3 source venv/bin/activate
``` 

or use the venv.sh script to acivate the environment and create an up to date directory map of your project:

```bash
source venv.sh
``` 
At the end of your session deactivate the venv: 

```bash
deactivate
``` 

**CUDA Support**

DinoCut is designed to work with [CUDA](https://pytorch.org/get-started/locally/) given it's reliance on [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO/tree/main) and [SAM](https://github.com/facebookresearch/segment-anything). To use CUDA you will need to ensure that your environment variable `CUDA_HOME` is set. 

Please make sure following the installation steps, 
 
#### Check CUDA_HOME Path:
```bash
echo $CUDA_HOME
```
If it print nothing, then it means you haven't set up the path/

Run this so the environment variable will be set under current shell. 
```bash
export CUDA_HOME=/path/to/cuda-11.8
```
In this example, /path/to/cuda-11.8 should be replaced with the path where your CUDA toolkit is installed. You can find this by typing **which nvcc** in your terminal:

For instance, 
if the output is /usr/local/cuda/bin/nvcc, then:
```bash
export CUDA_HOME=/usr/local/cuda/bin/nvcc/cuda-11.8
```

Notice the version of cuda should be aligned with your CUDA runtime in case you have multiple CUDA version installed. 

If you want to set the CUDA_HOME permanently, store it using:

```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
```
after that, source the bashrc file and check CUDA_HOME:
```bash
source ~/.bashrc
echo $CUDA_HOME
```

**Trouble Shooting CUDA:**

https://github.com/IDEA-Research/GroundingDINO/issues/193 



 # 📓 Resources & Inspiration # 

 [ImageBot](https://www.sciencedirect.com/science/article/pii/S2212827122002876?ref=pdf_download&fr=RR-2&rr=87b01ff6f8558bb9): A cut paste learn approach approach developed by Block et. al (2022). [Github Repo](https://github.com/FraunhoferIAO/Image-Bot)


# 📂 Directory structure 📂

    .
    ├── BlenderProc             # directory contains scripts, assets, and packages for synthetic data generation from 3D models
    ├── data                    # directory contains images, masks, and other assets for synthetic data generation
    ├── dataset                 # directory contains the final dataset containing images and YOLO-style labels for training the Neural Network
    ├── runs                    # directory contains the results of YOLO detections 
    ├── chroma.py               # a python script for generating synthetic datasets using a green screen.
    ├── requirements.txt	# latest PIP dependencies 	
    └── synthetic.py		# Script for generating the synthetic dataset using a CLI 	
    └── README.md

# Installation 

# Languages & Dependencies 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![Vim](https://img.shields.io/badge/VIM-%2311AB00.svg?style=for-the-badge&logo=vim&logoColor=white)

