# Predict-Numbers

A Convolutional Neural Network (CNN) model for handwritten digit recognition using the MNIST dataset.

## Model Performance
- Test Accuracy: 98.65%
- Test Loss: 0.0577


## Architecture
```
Model: CNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer                 Output Shape          Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conv2D               (None, 26, 26, 32)    320
MaxPooling2D         (None, 13, 13, 32)    0
Flatten              (None, 5408)          0
Dense                (None, 64)            346,176
Dense                (None, 10)            650
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total params: 347,146 (1.32 MB)
```

## Requirements
```python
tensorflow==2.17.1
numpy
pandas
matplotlib
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```python
python train.py
```

3. Make predictions:
```python
from tensorflow.keras.models import load_model

model = load_model("mnist_cnn_model.h5")
predictions = model.predict(x_test)
```

## Training Results
- Training Duration: 10 epochs
- Batch Size: 32
- Final Training Accuracy: 99.83%
- Final Validation Accuracy: 98.72%

## Model Features
- Convolutional layers for feature extraction
- MaxPooling for dimensionality reduction
- Dense layers for classification
- Adam optimizer
- Categorical crossentropy loss
