# Brain Tumor Detection using CNN

This project uses a Convolutional Neural Network (CNN) to detect brain tumors from MRI images. It is trained on real-world data from Kaggle and built using TensorFlow/Keras.

## Dataset

- Source: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Classes:
  - `yes/` → Images with tumor
  - `no/` → Images without tumor

## Model Overview

- Input size: 128x128 RGB images
- Architecture:
  - 3 Convolutional layers + MaxPooling
  - Flatten → Dense → Dropout → Output
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Accuracy: ~XX% (fill in after training)

## How to Run

1. Upload the dataset to Colab (or local)
2. Run `BrainTumorDetection.ipynb`
3. Train the model or load the saved `.h5` model
4. Use the model to make predictions on MRI images

## Trained Model Download

The trained model is too large for GitHub. You can download it here:

👉 [Download brain_tumor_model.h5](https://drive.google.com/file/d/1nNlgHUxuRyuVof3nVVGz2cbyON1nn771/view?usp=drive_link)

To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('brain_tumor_model.h5')

Requirements
Install dependencies using:
                   pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow



