# Manufacturing-Defect-Detection-DeepLearning-CNN

In this project, a deep learning classificationmodel using Convolutional Neural Networks (CNNs) is built to automate defect detection in products manufactured by casting. It improves quality control and redues costs.

---

## Project Overview
Quality control is a crucial part of manufacturing. Traditional defect detection methods rely on manual inspection, which is time-consuming, inconsistent, and costly. In this project, I apply **Deep Learning with Convolutional Neural Networks (CNNs)** to automate defect detection in manufactured products by casting process. The goal is to build an AI-powered system that can distinguish between **defective** and **non-defective** parts using image data.

---

## Problem Statement
- **Given product images from the manufacturing process, classify whether the product is defective or defect-free.**
- This problem is formulated as a **binary image classification problem**.

---

## Tools & Technique

- **Programming Language:** Python
- **Python Libraries:** Numpy, Pandas, Pytorch, optuna
- **Tool:** Google Colab with GPU, Git/GitHub
- **Technique:**
  - Deep Learning - Convolutional Neural Networks (CNN)
---

## Dataset

- The dataset contains 7,348 images labeled as defective or non-defective.
- All images are top-view photographs of submersible pump impellers, manufactured by casting.
- Images are 300×300 pixels, grayscale, with data augmentation already applied.
-  The dataset is organized into two main folders: `training` and `testing`. Each folder contains two subfolders: `ok_front` (non-defective) and `def_front` (defective).
---

## Model Approach ?? 
### Preprocessing
- Resize images for uniform input size
- Turn images to greyscale 
- Normalize pixel values

### CNN Architecture ???
- **Convolutional Layers** (Conv2D + ReLU) → extract spatial features  
- **Pooling Layers** (MaxPooling) → reduce dimensionality  
- **Flatten + Dense Layers** → learn feature interactions  
- **Dropout** → prevent overfitting  
- **Output Layer**: Sigmoid activation for binary classification  

### Training Setup ??
- Loss: Binary Cross-Entropy  
- Optimizer: Adam  
- Batch size: 32 / 64  
- Epochs: 20–50 (with early stopping)  

---

## Evaluation Metrics ???
- **Accuracy** : 0.9287  
- **Precision & Recall** : 0.9105
- **Recall** : 0.8931  
- **F1-score** : 0.9017  
- **Confusion Matrix** – defect vs non-defect predictions
<img width="513" height="470" alt="Confusion_matrix" src="https://github.com/user-attachments/assets/f0bd085b-c3e9-4026-a95a-8d36363b1597" />

## Classification Report
|                     | Precision | Recall    |f1-score   | support   |
|---------------------|-----------|-----------|-----------|-----------|
| `Non-Defective`     | 0.94      | 0.95      | 0.94      | 453       |    
| `Defective`         | 0.91      | 0.89      | 0.90      | 262       |
| `         `         |           |           |           |           |
| `accuracy`          |           |           | 0.93      | 715       |
| `macro avg`         | 0.92      | 0.92      | 0.92      | 715       |
| `weighted avg`      | 0.93      | 0.93      | 0.93      | 715       |

---

## Project Structure 

```
Manufacturing-Defect-Detection-DeepLearning-CNN/
├── data/
│ ├── train/
│ │   ├── def_front/       #Images containing defect 
│ │   └── ok_front/        #Images with no defect
│ └── test/           
│     ├── def_front/       #Images containing defect
│     └── ok_front/        #Images with no defect
├── jupyter_notebook_Script/
│     └── Manufacturing_Defect_Detection_DeepLearning_CNN.ipynb
├── output/
│     └── Manufacturing_Defect_Detection_DeepLearning_CNN.docx
├── models/
│     ├── best_model.pth
│     ├── full_model.pth
│     ├── model_weights.pth
│     └── checkpoint.pth
└── README.md

```

---
## Business Value
- **Automated defect detection** → faster, cheaper, and more consistent than manual inspection  
- **Improved quality control** → reduces waste, rework, and warranty costs  
- **Scalable AI solution** → applicable across different manufacturing lines and products  
---
## Future Work
- Extend the model to multi-class classification to identify different types of manufacturing defects.
- Develop a REST API and deploy the model to the cloud (AWS) for real-time, online defect detection.
- Integrate with production pipelines to enable automated, scalable quality control.
- Explore advanced CNN architectures (e.g., ResNet, EfficientNet) to improve accuracy and robustness.
---

## Acknowledgments
- Dataset: [Casting Product Defect Dataset (Kaggle)](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
---

## Author

**Abolfazl Zolfaghari**
[Email](ab.zolfaghari.abbasghaleh) | [GitHub](https://github.com/abolfazl6678)

---
