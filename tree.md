.
├── DINOcut
│   ├── __init__.py
│   └── dinocut.py
├── Dockerfile
├── GroundingDINO
│   ├── Dockerfile
│   ├── LICENSE
│   ├── README.md
│   ├── build
│   │   ├── lib.linux-x86_64-3.10
│   │   │   └── groundingdino
│   │   │       ├── _C.cpython-310-x86_64-linux-gnu.so
│   │   │       ├── __init__.py
│   │   │       ├── config
│   │   │       │   ├── GroundingDINO_SwinB_cfg.py
│   │   │       │   ├── GroundingDINO_SwinT_OGC.py
│   │   │       │   └── __init__.py
│   │   │       ├── datasets
│   │   │       │   ├── __init__.py
│   │   │       │   ├── cocogrounding_eval.py
│   │   │       │   └── transforms.py
│   │   │       ├── models
│   │   │       │   ├── GroundingDINO
│   │   │       │   │   ├── __init__.py
│   │   │       │   │   ├── backbone
│   │   │       │   │   │   ├── __init__.py
│   │   │       │   │   │   ├── backbone.py
│   │   │       │   │   │   ├── position_encoding.py
│   │   │       │   │   │   └── swin_transformer.py
│   │   │       │   │   ├── bertwarper.py
│   │   │       │   │   ├── fuse_modules.py
│   │   │       │   │   ├── groundingdino.py
│   │   │       │   │   ├── ms_deform_attn.py
│   │   │       │   │   ├── transformer.py
│   │   │       │   │   ├── transformer_vanilla.py
│   │   │       │   │   └── utils.py
│   │   │       │   ├── __init__.py
│   │   │       │   └── registry.py
│   │   │       ├── util
│   │   │       │   ├── __init__.py
│   │   │       │   ├── box_ops.py
│   │   │       │   ├── get_tokenlizer.py
│   │   │       │   ├── inference.py
│   │   │       │   ├── logger.py
│   │   │       │   ├── misc.py
│   │   │       │   ├── slconfig.py
│   │   │       │   ├── slio.py
│   │   │       │   ├── time_counter.py
│   │   │       │   ├── utils.py
│   │   │       │   ├── visualizer.py
│   │   │       │   └── vl_utils.py
│   │   │       └── version.py
│   │   └── temp.linux-x86_64-3.10
│   │       └── home
│   │           └── nalkuq
│   │               └── cmm_article
│   │                   └── GroundingDINO
│   │                       └── groundingdino
│   │                           └── models
│   │                               └── GroundingDINO
│   │                                   └── csrc
│   │                                       ├── MsDeformAttn
│   │                                       │   ├── ms_deform_attn_cpu.o
│   │                                       │   └── ms_deform_attn_cuda.o
│   │                                       ├── cuda_version.o
│   │                                       └── vision.o
│   ├── demo
│   │   ├── create_coco_dataset.py
│   │   ├── gradio_app.py
│   │   ├── image_editing_with_groundingdino_gligen.ipynb
│   │   ├── image_editing_with_groundingdino_stablediffusion.ipynb
│   │   ├── inference_on_a_image.py
│   │   └── test_ap_on_coco.py
│   ├── docker_test.py
│   ├── environment.yaml
│   ├── groundingdino
│   │   ├── _C.cpython-310-x86_64-linux-gnu.so
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-310.pyc
│   │   ├── config
│   │   │   ├── GroundingDINO_SwinB_cfg.py
│   │   │   ├── GroundingDINO_SwinT_OGC.py
│   │   │   └── __init__.py
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   └── transforms.cpython-310.pyc
│   │   │   ├── cocogrounding_eval.py
│   │   │   └── transforms.py
│   │   ├── models
│   │   │   ├── GroundingDINO
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   ├── bertwarper.cpython-310.pyc
│   │   │   │   │   ├── fuse_modules.cpython-310.pyc
│   │   │   │   │   ├── groundingdino.cpython-310.pyc
│   │   │   │   │   ├── ms_deform_attn.cpython-310.pyc
│   │   │   │   │   ├── transformer.cpython-310.pyc
│   │   │   │   │   ├── transformer_vanilla.cpython-310.pyc
│   │   │   │   │   └── utils.cpython-310.pyc
│   │   │   │   ├── backbone
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── __pycache__
│   │   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   │   ├── backbone.cpython-310.pyc
│   │   │   │   │   │   ├── position_encoding.cpython-310.pyc
│   │   │   │   │   │   └── swin_transformer.cpython-310.pyc
│   │   │   │   │   ├── backbone.py
│   │   │   │   │   ├── position_encoding.py
│   │   │   │   │   └── swin_transformer.py
│   │   │   │   ├── bertwarper.py
│   │   │   │   ├── csrc
│   │   │   │   │   ├── MsDeformAttn
│   │   │   │   │   │   ├── ms_deform_attn.h
│   │   │   │   │   │   ├── ms_deform_attn_cpu.cpp
│   │   │   │   │   │   ├── ms_deform_attn_cpu.h
│   │   │   │   │   │   ├── ms_deform_attn_cuda.cu
│   │   │   │   │   │   ├── ms_deform_attn_cuda.h
│   │   │   │   │   │   └── ms_deform_im2col_cuda.cuh
│   │   │   │   │   ├── cuda_version.cu
│   │   │   │   │   └── vision.cpp
│   │   │   │   ├── fuse_modules.py
│   │   │   │   ├── groundingdino.py
│   │   │   │   ├── ms_deform_attn.py
│   │   │   │   ├── transformer.py
│   │   │   │   ├── transformer_vanilla.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   └── registry.cpython-310.pyc
│   │   │   └── registry.py
│   │   ├── util
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── box_ops.cpython-310.pyc
│   │   │   │   ├── get_tokenlizer.cpython-310.pyc
│   │   │   │   ├── inference.cpython-310.pyc
│   │   │   │   ├── misc.cpython-310.pyc
│   │   │   │   ├── slconfig.cpython-310.pyc
│   │   │   │   ├── utils.cpython-310.pyc
│   │   │   │   ├── visualizer.cpython-310.pyc
│   │   │   │   └── vl_utils.cpython-310.pyc
│   │   │   ├── box_ops.py
│   │   │   ├── get_tokenlizer.py
│   │   │   ├── inference.py
│   │   │   ├── logger.py
│   │   │   ├── misc.py
│   │   │   ├── slconfig.py
│   │   │   ├── slio.py
│   │   │   ├── time_counter.py
│   │   │   ├── utils.py
│   │   │   ├── visualizer.py
│   │   │   └── vl_utils.py
│   │   └── version.py
│   ├── groundingdino.egg-info
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── requires.txt
│   │   └── top_level.txt
│   ├── requirements.txt
│   ├── setup.py
│   ├── test.ipynb
│   └── weights
│       └── groundingdino_swint_ogc.pth
├── LICENSE
├── README.Docker.md
├── README.md
├── assets
│   ├── DINOcut fig 1.png
│   ├── DINOcut_thumbnail.png
│   ├── dinocut_video_thumbnail.png
│   └── examples
│       └── images
│           ├── Original.jpg
│           ├── beach spongebob caveman meme.png
│           ├── einstein_training_dataset.png
│           ├── spongebob caveman meme.jpg
│           └── spongebob mocking meme.jpg
├── citation.cff
├── compose.yaml
├── data
│   ├── background
│   │   └── h_0a9a3552.jpg
│   ├── bg_noise
│   │   ├── images
│   │   │   ├── 1.png
│   │   │   ├── 2.jpg
│   │   │   ├── 3.jpg
│   │   │   ├── 4.jpg
│   │   │   └── 5.jpg
│   │   └── masks
│   │       ├── 1.png
│   │       ├── 2.png
│   │       ├── 3.jpg
│   │       ├── 4.png
│   │       └── 5.png
│   └── spongebob squarepants
│       ├── images
│       │   └── mask_1718825050.3183672.jpg
│       └── masks
│           └── mask_1718825050.3183672.png
├── dataset
│   ├── test
│   │   ├── images
│   │   │   ├── 02024_06_19-03_24_18_PM.jpg
│   │   │   ├── 102024_06_19-03_24_18_PM.jpg
│   │   │   ├── 112024_06_19-03_24_18_PM.jpg
│   │   │   ├── 12024_06_19-03_24_18_PM.jpg
│   │   │   ├── 122024_06_19-03_24_18_PM.jpg
│   │   │   ├── 132024_06_19-03_24_18_PM.jpg
│   │   │   ├── 142024_06_19-03_24_18_PM.jpg
│   │   │   ├── 152024_06_19-03_24_18_PM.jpg
│   │   │   ├── 162024_06_19-03_24_18_PM.jpg
│   │   │   ├── 172024_06_19-03_24_18_PM.jpg
│   │   │   ├── 182024_06_19-03_24_18_PM.jpg
│   │   │   ├── 192024_06_19-03_24_18_PM.jpg
│   │   │   ├── 202024_06_19-03_24_18_PM.jpg
│   │   │   ├── 212024_06_19-03_24_18_PM.jpg
│   │   │   ├── 22024_06_19-03_24_18_PM.jpg
│   │   │   ├── 222024_06_19-03_24_18_PM.jpg
│   │   │   ├── 232024_06_19-03_24_18_PM.jpg
│   │   │   ├── 242024_06_19-03_24_18_PM.jpg
│   │   │   ├── 32024_06_19-03_24_18_PM.jpg
│   │   │   ├── 42024_06_19-03_24_18_PM.jpg
│   │   │   ├── 52024_06_19-03_24_18_PM.jpg
│   │   │   ├── 62024_06_19-03_24_18_PM.jpg
│   │   │   ├── 72024_06_19-03_24_18_PM.jpg
│   │   │   ├── 82024_06_19-03_24_18_PM.jpg
│   │   │   └── 92024_06_19-03_24_18_PM.jpg
│   │   └── labels
│   │       ├── 02024_06_19-03_24_18_PM.txt
│   │       ├── 102024_06_19-03_24_18_PM.txt
│   │       ├── 112024_06_19-03_24_18_PM.txt
│   │       ├── 12024_06_19-03_24_18_PM.txt
│   │       ├── 122024_06_19-03_24_18_PM.txt
│   │       ├── 132024_06_19-03_24_18_PM.txt
│   │       ├── 142024_06_19-03_24_18_PM.txt
│   │       ├── 152024_06_19-03_24_18_PM.txt
│   │       ├── 162024_06_19-03_24_18_PM.txt
│   │       ├── 172024_06_19-03_24_18_PM.txt
│   │       ├── 182024_06_19-03_24_18_PM.txt
│   │       ├── 192024_06_19-03_24_18_PM.txt
│   │       ├── 202024_06_19-03_24_18_PM.txt
│   │       ├── 212024_06_19-03_24_18_PM.txt
│   │       ├── 22024_06_19-03_24_18_PM.txt
│   │       ├── 222024_06_19-03_24_18_PM.txt
│   │       ├── 232024_06_19-03_24_18_PM.txt
│   │       ├── 242024_06_19-03_24_18_PM.txt
│   │       ├── 32024_06_19-03_24_18_PM.txt
│   │       ├── 42024_06_19-03_24_18_PM.txt
│   │       ├── 52024_06_19-03_24_18_PM.txt
│   │       ├── 62024_06_19-03_24_18_PM.txt
│   │       ├── 72024_06_19-03_24_18_PM.txt
│   │       ├── 82024_06_19-03_24_18_PM.txt
│   │       └── 92024_06_19-03_24_18_PM.txt
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
├── dinocut.py
├── dinocut_config.yaml
├── docs
├── requirements.txt
├── sam
│   └── weights
│       └── sam_vit_h_4b8939.pth
├── scripts
│   ├── chroma.py
│   ├── figures.py
│   ├── selector.py
│   ├── synthetic.py
│   ├── synthetic_config.yaml
│   ├── tik_tok_scraper.py
│   └── visualize.py
├── setup.py
├── starter_dataset
│   └── spongebob mocking meme.jpg
├── test
│   ├── args_spongebob.yaml
│   ├── args_ulu.yaml
│   ├── dataset_check.py
│   ├── log.txt
│   ├── log_spongebob.txt
│   ├── log_uluaq.txt
│   ├── results_spongebob.csv
│   └── results_ulu.csv
├── tree.md
└── venv.sh

66 directories, 230 files
