# RD-Net


RD-NET: A Deep Dilated-Receptive Residual Block-Based Network for Multi-class bone segmentation of Hip joint CT Scans.

# Overall Structure of RD-Net
![alt text](https://github.com/Siddhesh6344/RD-Net/blob/main/Model%20Architecture.png)

# Qualitative comparisions of Results between different models
![image](https://github.com/user-attachments/assets/33818670-a1a0-48e4-9f13-3b22746272de)

# Installation
Requirements:

```bash
pip install -r requirements.txt
```

The trained weights have been released (see below)

# Input
It is expected that all inputs should be in volumetric format for data like CT scans and MRI images that handles co-registration and are correctly aligned to a common reference.

# Instructions for Running the Model

# 1. Clone the repository
```bash
$ git clone https://github.com/Siddhesh6344/RD-Net.git
```

1. Data Preparation: Our model is compatible with volumeteric medical imaging datasets like CT scans, MRIs, .etc. We are using CTPEL dataset from AIDA Datahub which has Hip joint CT scans of 90 patients consisting of 5 bone segmentation masks and 15 anatomical landmarks for pelvis bones. CTPEL dataset can be obtained from license sharing agreement with AIDA Datahub from author's approval. (Copyright 2019 KTH, Chunliang Wang). (if neccessary you can use any other volumeteric CT or MRI dataset) 

2. Experimental Environment: We recommend to use CUDA supported NVIDIA graphic cards for running purposes. We are running our current setup on PARAM Shakti Server system with 4 NVIDIA V100 GPUs each having a VRAM of 32GB. Please install the specific versions of libraries from requirements.txt. Please install the required python libraries before use, and please install git and add it to the system environment variable so that you can use the bash command.

3. 
