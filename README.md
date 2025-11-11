# Facial Keypoints Detection Submission

A deep learning project that uses Convolutional Neural Networks (CNN) to detect and predict facial keypoints in images. This project was developed for the Kaggle Facial Keypoints Detection competition.

## 🎯 Project Overview

This project implements a CNN-based solution to predict the location of 15 keypoints on face images. The keypoints include important facial features such as:
- Eye corners and centers
- Eyebrow points
- Nose tip
- Mouth corners and center

## 📋 Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Kaggle API

## 🚀 Installation & Setup

1. **Install required packages:**
```bash
pip install kaggle tensorflow pandas numpy scikit-learn matplotlib
```

2. **Setup Kaggle API:**
   - Download your `kaggle.json` API key from your Kaggle account
   - Place it in the `~/.kaggle/` directory
   - Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the dataset:**
```bash
kaggle competitions download -c facial-keypoints-detection
```

## 📁 Dataset Structure

The project uses the Kaggle Facial Keypoints Detection dataset which includes:
- `training.csv` - Training images and keypoint coordinates
- `test.csv` - Test images for prediction
- `IdLookupTable.csv` - Mapping table for submission format
- `SampleSubmission.csv` - Sample submission file format

## 🏗️ Model Architecture

The CNN model consists of:
- **3 Convolutional Blocks** with BatchNormalization and MaxPooling:
  - Conv2D(32) → BatchNorm → MaxPool
  - Conv2D(64) → BatchNorm → MaxPool  
  - Conv2D(128) → BatchNorm → MaxPool
- **Dense Layers:**
  - Flatten layer
  - Dense(512) with ReLU activation
  - Dropout(0.2) for regularization
  - Dense(30) output layer (for 15 keypoints × 2 coordinates)

## 🔧 Key Features

### Data Preprocessing
- **Missing Data Handling:** Removes rows with missing keypoint values
- **Image Normalization:** Converts pixel values to range [0, 1]
- **Data Reshaping:** Converts string pixel data to 96×96 grayscale images

### Model Training
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Metrics:** Mean Absolute Error (MAE)
- **Regularization:** Dropout and Early Stopping
- **Validation Split:** 80/20 train/validation split

### Training Configuration
- **Epochs:** Up to 100 (with early stopping)
- **Batch Size:** 128
- **Early Stopping:** Patience of 10 epochs
- **Validation:** 20% of training data

## 📊 Model Performance Visualization

The script generates training visualizations including:
- Training vs Validation Loss curves
- Training vs Validation MAE curves

## 🔍 Prediction Visualization

The model includes functionality to:
- Make predictions on test images
- Clip predictions to valid coordinate range [0, 96]
- Visualize predicted keypoints overlaid on test images
- Display random sample predictions for validation

## 📤 Submission Generation

The final step creates a Kaggle-compatible submission file:
1. Makes predictions on all test images
2. Maps predictions using the IdLookupTable
3. Generates `submission.csv` in the required format

## 🏃‍♂️ Usage

1. **Prepare the environment:**
```python
# Upload kaggle.json and run the setup cells
```

2. **Download and extract data:**
```python
# The script automatically downloads and extracts the competition data
```

3. **Train the model:**
```python
# Run the training cells - model will train with early stopping
```

4. **Generate predictions:**
```python
# The script will create submission.csv automatically
```

## 📈 Results

- The model uses MSE loss for training
- MAE is tracked as an additional metric
- Early stopping prevents overfitting
- Predictions are clipped to ensure valid coordinate ranges
- Final submission file is generated in Kaggle format

## 🛠️ Model Improvements

Potential enhancements for better performance:
- **Data Augmentation:** Rotation, flipping, scaling
- **Advanced Architectures:** ResNet, DenseNet, or custom architectures
- **Ensemble Methods:** Combining multiple models
- **Hyperparameter Tuning:** Grid search or Bayesian optimization
- **Transfer Learning:** Using pre-trained models

## 📝 Notes

- Images are 96×96 pixels in grayscale
- The model predicts 30 values (15 keypoints × 2 coordinates each)
- Missing data is handled by dropping incomplete samples
- Predictions are automatically clipped to valid pixel ranges

## 🤝 Contributing

Feel free to fork this project and submit improvements via pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Competition:** [Kaggle Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection)

**Author:** Timniel
