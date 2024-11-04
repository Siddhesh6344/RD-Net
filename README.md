# RD-Net


RD-NET: A Deep Dilated-Receptive Residual Block-Based Network for Multi-class bone segmentation of Hip joint CT Scans.


# Overall structure of RD-Net
![alt text](https://github.com/Siddhesh6344/RD-Net/blob/main/Model%20Architecture.png)

# Qualitative comparisions of Results between different models
![image](https://github.com/user-attachments/assets/33818670-a1a0-48e4-9f13-3b22746272de)


# Instructions for Running the Model

1. Data Preparation: Our model is compatible with volumeteric medical imaging datasets like CT scans, MRIs, .etc. We are using CTPEL dataset from AIDA Datahub which has Hip joint CT scans of 90 patients consisting of 5 bone segmentation masks and 15 anatomical landmarks for pelvis bones. CTPEL dataset can be obtained from license sharing agreement with AIDA Datahub from author's approval. (Copyright 2019 KTH, Chunliang Wang).

2. Experimental Environment: We recommend to use CUDA supported NVIDIA graphic cards for running purposes. We are running our current setup on PARAM Shakti Server system with 4 NVIDIA V100 GPUs each having a VRAM of 32GB. Please install the specific versions of libraries from requirements.txt.
