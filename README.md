## Surface Crack Detection Using Convolutional Neural Networks

### Overview

This project focuses on automated surface crack detection from concrete images using convolutional neural networks (CNNs). The goal is to evaluate multiple deep learning architectures for binary classification (Cracked vs. Non-Cracked) and identify practical strengths, limitations, and trade-offs without overstating performance.

The work is motivated by the need to reduce manual inspection effort in civil infrastructure and manufacturing, where missed cracks can lead to safety risks and increased maintenance costs.

---

#### VGG16 (Best Performing Model)
![img](https://github.com/nandhika03/surfacecrackdetection-pyspark/blob/main/Reports/VGG16OUTPUT.png) ![img](https://github.com/nandhika03/surfacecrackdetection-pyspark/blob/main/Reports/vggbestmodel.png)

* Overall accuracy: **0.97**
* Precision (Cracked): **0.95–0.99**
* Recall (Cracked): **0.95–0.99**
* Very low misclassification rate for cracked surfaces

---
### Problem Statement

Manual inspection of concrete surfaces is time-consuming and prone to human error. This project explores whether image-based deep learning models can reliably distinguish cracked from non-cracked surfaces using labeled image data.

**Task:** Binary image classification

* Class 0: Non-Cracked
* Class 1: Cracked

---

### Dataset

* Source: Kaggle (Concrete Crack Images dataset)
* Total images: **40,000**
* Cracked images: **20,000**
* Non-cracked images: **20,000**
* Image sizes: resized during preprocessing

#### Data Split

* Training: **70% (28,000 images)**
* Validation: **15% (6,000 images)**
* Test: **15% (6,000 images)**

---

### Models Evaluated

The following architectures were implemented and evaluated:

* Custom CNN
* LeNet
* VGG16 (transfer learning)
* ResNet50 (transfer learning)

Training and experimentation were performed using **Python** in **Google Colab**, with standard deep learning libraries.

---

### Quantum-Inspired CNN (QCNN)

A custom CNN architecture incorporating batch normalization and dropout was referred to as a *Quantum-Inspired CNN* due to its hierarchical feature processing and efficient dimensionality reduction. The model does **not** use true quantum computation and is implemented entirely using classical deep learning methods.

#### Input

* Image size: **64 × 64 × 3**
* Pixel values normalized to [0, 1]

#### Architecture Summary

* Conv2D (32 filters, 3×3) + BatchNorm + MaxPooling
* Conv2D (64 filters, 3×3) + BatchNorm + MaxPooling
* Flatten
* Dense (128) + Dropout (0.5)
* Dense (64) + Dropout (0.5)
* Output Dense (2, Softmax)

---

### Hyperparameters (Common Across Models)

* Optimizer: Adam
* Batch size: 16–64 (model dependent)
* Dropout rate: 0.5
* Epochs: 4–10
* Loss function: Categorical Cross-Entropy

---

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Metrics are reported on the **held-out test set**.

---

### Key Results (Test Set)

#### QCNN (Custom CNN)

* Accuracy: **0.78**
* Precision (Cracked): **0.57**
* Recall (Cracked): **1.00**
* F1-score (Cracked): **0.72**

The model shows strong sensitivity to cracks but poor discrimination of non-cracked surfaces, indicating class bias.

#### CNN

* Accuracy: **0.98**
* Precision: **0.96**
* Recall: **1.00**
* F1-score: **0.98**

#### LeNet

* Accuracy: **0.96–0.97**
* Precision: **0.95**
* Recall: **0.97**
* F1-score: **0.97**


#### ResNet50

* Stable performance with higher computational cost
* No clear improvement over VGG16 for this dataset

---

### Model Selection
![img](https://github.com/nandhika03/surfacecrackdetection-pyspark/blob/main/Reports/VGG16Workflow.png)
**VGG16** was selected as the best-performing model based on:
* High recall for cracked surfaces
* Low number of false negatives
* Stable generalization compared to deeper architectures

---

### Limitations

* Validation accuracy drops in some models indicate **overfitting**.
* Performance is measured on a curated dataset; real-world images may include noise, lighting variation, and surface artifacts.
* The QCNN model shows class bias and requires further tuning.
* No crack localization or segmentation is performed—classification only.
* Computational time and memory usage were not formally benchmarked.

---

### Future Work

* Crack localization using segmentation models (e.g., U-Net)
* Stronger regularization and early stopping
* Cross-dataset validation
* Lightweight deployment-ready models

---

### Project Structure

```
├── code/        # Google Colab notebooks and Python scripts
├── report/      # Final report, figures, and results
└── README.md
```

---

### License

This project is released under the **MIT License**. The dataset remains subject to Kaggle’s original licensing terms.

---

### Team

* Nandhika Rajmanikandan
* Ganesh Vannam
* Madhumitha Mandyam
* Muneendra Magani


