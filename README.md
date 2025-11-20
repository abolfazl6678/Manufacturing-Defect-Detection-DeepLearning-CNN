# Manufacturing-Defect-Detection-DeepLearning-CNN

In this project, a deep learning model using Convolutional Neural Networks (CNNs) is built to automate defect detection in products manufactured by casting. It improves quality control and redues costs. Casting Product Defect Dataset from Kaggle is used.

---

## Project Overview
Quality control is a crucial part of manufacturing. Traditional defect detection methods rely on manual inspection, which is time-consuming, inconsistent, and costly. In this project, I apply **Deep Learning with Convolutional Neural Networks (CNNs)** to automate defect detection in manufactured products. The goal is to build an AI-powered system that can distinguish between **defective** and **non-defective** parts using image data.

---

## Problem Statement ?? 
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

## Dataset & Variables??? 
The mode is built based on dataset provided in Kaggle (see Acknowledgments section please). It consists of two 


Each dataset contains labeled images of **good** and **defective** products, ideal for training CNNs.  


The dataset was provided in Kaggle (see Acknowledgments section) and includes detailed of retail store inventory as a table named **retail_store_inventory** with below varaibles.








---

## Model Approach ?? 
### Preprocessing
- Resize images for uniform input size  
- Normalize pixel values  
- Data augmentation (rotation, flips, contrast adjustment) to improve generalization  

### CNN Architecture
- **Convolutional Layers** (Conv2D + ReLU) → extract spatial features  
- **Pooling Layers** (MaxPooling) → reduce dimensionality  
- **Flatten + Dense Layers** → learn feature interactions  
- **Dropout** → prevent overfitting  
- **Output Layer**: Sigmoid activation for binary classification  

### Training Setup
- Loss: Binary Cross-Entropy  
- Optimizer: Adam  
- Batch size: 32 / 64  
- Epochs: 20–50 (with early stopping)  

---

## Evaluation Metrics ???
- **Accuracy** – overall correctness  
- **Precision & Recall** – crucial for quality control (minimizing false negatives)  
- **F1-score** – balance of precision and recall  
- **Confusion Matrix** – defect vs non-defect predictions  

---

## Results (to be updated after training) ???
- Model accuracy and loss curves  
- Confusion matrix and classification report  
- Visualizations of predictions with Grad-CAM heatmaps (to explain CNN decisions)  

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
