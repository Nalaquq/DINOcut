
<div align="center">
    <img src="./assets/DINOcut_thumbnail.png" width="%">
</div>


# 🦖 DinoCut ✂️


[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=Cf0wft5CKT4) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xbq9rEOtyQh8QUQU-__E-Ub3Wy3X1NoV)[![Static Badge](https://img.shields.io/badge/GroundingDINO-arXiv-blue)](https://arxiv.org/abs/2303.05499) [![Static Badge](https://img.shields.io/badge/Segment_Anything-arXiv-blue)](https://arxiv.org/abs/2304.02643) [![Static Badge](https://img.shields.io/badge/Cut_Paste_Learn-arXiv-blue)](https://arxiv.org/abs/1708.01642) [![Static Badge](https://img.shields.io/badge/Grounded_SAM-arXiv-blue)](https://arxiv.org/abs/2401.14159)



[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)


# 🧠 🚀Conceptual Design 📚 🤯
The goal of this project is simple: to combine sommething old with soemthing new. So we've created an image processing pipeline for object detection using Grounding DINO; SAM; and a cut, paste learn (BGcut) approach. The result is a semi-supervised image processing pipeline that allows users to generate large, synthetic datasets for object detection without the hassle of manually labeling bounding boxes or creating segmentation masks. 
 
 # 📓 

# Directory structure

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

