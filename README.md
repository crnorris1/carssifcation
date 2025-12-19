# Vehicle Classification with PyTorch

![A demo gif of the website](car_classifier_demo.gif)

[**Hosted on Render!**](https://car-classifier-frontend.onrender.com/)

**NOTE:** Since we are using the free version of render, the backend may not load in time for the first request. If an upload fails, refresh and page and try again. 

## Overview
This project implements a deep learning–based vehicle classifier with a simple UI frontend. Users can upload a side-view image of a vehicle, and the model predicts whether it is a sedan, SUV, or pickup truck within seconds. The model was trained on real-world images containing significant background noise, requiring it to learn distinguishing visual features rather than relying on clean, cropped inputs.

## Dataset
Images were manually collected around the WPI campus and organized into class-specific folders to match PyTorch’s expected dataset structure.

- **Classes:** Sedan, SUV, Truck  
- **Train/Test Split:** 80% / 20%  
- **Final Dataset Size:**
  - 82 SUVs  
  - 44 Sedans  
  - 22 Trucks  

## Methods
- **Framework:** PyTorch  
- **Model:** Pretrained ResNet18 (transfer learning)  
- **Training:** NVIDIA GPU with CUDA  
- **Data Augmentation:** Random horizontal flipping  
- **Normalization:** Standard image normalization  
- **Optimization:** Tuned batch size, dataloader parameters, and class-weighted loss to reduce truck misclassification  

## Results
Model performance improved significantly as more data was collected and training parameters were optimized. The best-performing configuration achieved approximately **97% testing accuracy**, although can most likely be attributed to overfitting due to the relatively small dataset.

## Conclusions
This project demonstrates how transfer learning and careful data handling can produce strong results even with a relatively small dataset. While the model performs well overall, collecting additional and more diverse data, especially for underrepresented classes like pickup trucks, would further improve generalization and robustness.
