# Beyond Visible Spectrum AI for Agriculture

## Author
BUKASA MUYOMBO 

## Project Overview
A deep learning model for detecting plant diseases using spectral imaging, incorporating HOG (Histogram of Oriented Gradients) and LBP (Local Binary Pattern) features for enhanced disease classification.

## Features
- Spectral image processing and normalization
- Multiple feature extraction methods:
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Pattern)
- Data augmentation techniques
- CNN-based classification
- Model conversion to TFLite for mobile deployment

## Model Architecture
- Convolutional Neural Network (CNN) with:
  - Multiple Conv2D layers
  - Batch Normalization
  - MaxPooling
  - Dropout layers for regularization
  - Dense layers for classification

## Dataset Structure
archive/
├── train/
│ ├── health/
│ ├── other/
│ └── rust/
└── val/
├── health/
├── other/
└── rust/

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```python
# Convert TIF images to PNG
convertTifToPNG(train_dir)
convertTifToPNG(val_dir)
```

### Training the Model
```python
# Train the model
model.fit(trainDataset, 
          validation_data=valDataset, 
          epochs=epochs, 
          callbacks=[earlyStopping, lrSCheduler, checkpoint])
```

### Making Predictions
```python
# Load and prepare image
img = image.load_img(imagePath, target_size=(256, 256))
# Make prediction
predictions = model.predict(imageArray)
```

## Model Performance Metrics
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training/Validation Accuracy
- Training/Validation Loss

## Visualizations
- Confusion Matrix
- Accuracy curves
- Loss curves
- Classification metrics

## Model Export
- Saves Keras model (.h5 format)
- Converts to TFLite for mobile deployment

## Features
1. **Image Processing**
   - TIF to PNG conversion
   - Spectral normalization
   - Grayscale conversion

2. **Feature Extraction**
   - HOG features
   - LBP features

3. **Data Augmentation**
   - Random flips
   - Random rotation
   - Pixel value normalization

4. **Model Training**
   - Early stopping
   - Learning rate scheduling
   - Model checkpointing

## Acknowledgments
- Dataset source: Beyond Visible Spectrum AI for Agriculture 2024
