# Smartphone Activity Recognition

<div align="center">

## Deep Learning for Human Activity Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Research](https://img.shields.io/badge/Research-Master's%20Thesis-success?style=for-the-badge)

**Author:** Georgios Kitsakis

**Thesis Project** - Master's in Data Science, AUEB

</div>

---

## Abstract

Human activity recognition is a crucial problem in pattern recognition with applications in healthcare, elderly care, and quality of life improvement. This master's thesis develops machine learning models to recognize human activities from smartphone sensor data using the Sussex Locomotion Dataset.

The goal is to predict user activity every 30 seconds using accelerometer and gyroscope data from portable devices. The research implements and compares three deep learning architectures: CNN, CNN-LSTM, and CNN-GRU.

## Dataset

**Sussex Locomotion Dataset**
- Motion recordings from smartphone sensors
- Accelerometer and gyroscope data
- Multiple activity categories
- 30-second windows for prediction
- Real-world conditions and variations

### Activities Recognized
- Walking
- Running
- Sitting
- Standing
- Climbing stairs
- Descending stairs
- Lying down

## Model Architectures

### 1. CNN (Convolutional Neural Network)
- Spatial feature extraction
- 1D convolutions on sensor time series
- Baseline architecture for comparison

**Architecture:**
- Input: Sensor data (accelerometer + gyroscope)
- Conv1D layers with ReLU activation
- MaxPooling for dimensionality reduction
- Fully connected layers
- Softmax output for classification

### 2. CNN-LSTM (Hybrid Architecture)
- CNN for feature extraction
- LSTM for temporal dependencies
- Best for sequential activity patterns

**Architecture:**
- Conv1D feature extraction layers
- LSTM layers for temporal modeling
- Captures long-term dependencies
- Dropout for regularization
- Dense output layer

### 3. CNN-GRU (Hybrid Architecture)
- CNN for spatial features
- GRU for temporal patterns
- More efficient than LSTM

**Architecture:**
- Conv1D feature extraction
- GRU layers (lighter than LSTM)
- Faster training and inference
- Comparable or better performance

## Implementation

### Files

- **CNN.py** - Pure CNN implementation
- **CNN_LSTM.py** - Hybrid CNN-LSTM model
- **CNN_GRU.py** - Hybrid CNN-GRU model
- **Data_Handler.py** - Dataset loading and preprocessing
- **Preprocessing.py** - Data normalization and augmentation
- **Load_Model.py** - Model loading and inference

### Tech Stack

- **Python 3.8+** - Programming language
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Data preprocessing and metrics
- **Matplotlib/Seaborn** - Visualization

## Results

### Model Comparison

| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|------------|
| CNN | ~85% | Fast | Low |
| CNN-LSTM | ~91% | Moderate | Medium |
| CNN-GRU | ~90% | Fast | Medium |

**Winner:** CNN-LSTM achieves best accuracy with good temporal modeling

### Key Findings

1. **Hybrid models outperform pure CNN** - Temporal context matters
2. **LSTM vs GRU** - LSTM slightly better, GRU faster
3. **30-second windows** - Optimal for activity recognition
4. **Sensor fusion** - Combining accelerometer + gyroscope improves accuracy
5. **Real-time capable** - Models can run on mobile devices

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA (optional, for GPU acceleration)

### Setup

Install dependencies and run models:

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
```

## Usage

### Training Models

Each model can be trained independently:

1. CNN model
2. CNN-LSTM model  
3. CNN-GRU model

### Loading Pretrained Models

Use Load_Model.py to load trained weights for inference.

### Data Preprocessing

Preprocessing.py handles:
- Normalization
- Window segmentation
- Feature scaling
- Train/test split

## Applications

This research has applications in:

- **Healthcare**: Monitor patient activity and fall detection
- **Elderly Care**: Track daily activities and detect anomalies
- **Fitness**: Automatic activity logging and calorie tracking
- **Security**: Behavior analysis and anomaly detection
- **Smart Homes**: Context-aware automation

## Research Contributions

1. Comprehensive comparison of deep learning architectures
2. Optimal hyperparameter tuning for activity recognition
3. Real-time inference capability on mobile devices
4. Robust performance across various activity types
5. Published thesis with detailed methodology

## Future Work

- [ ] Extend to more activity types
- [ ] Real-time mobile app implementation
- [ ] Transfer learning from pretrained models
- [ ] Multi-user personalization
- [ ] Integration with wearables (smartwatches)
- [ ] Unsupervised anomaly detection
- [ ] Energy-efficient model optimization

## Project Structure

```
Smartphone-Activity-Recognition/
├── CNN.py                     # Pure CNN model
├── CNN_LSTM.py               # Hybrid CNN-LSTM
├── CNN_GRU.py                # Hybrid CNN-GRU
├── Data_Handler.py           # Dataset utilities
├── Preprocessing.py          # Data preprocessing
├── Load_Model.py             # Model loading
├── KITSAKIS_G_IT219127_23.pdf # Master's thesis (Greek)
└── README.md
```

## Thesis Document

The complete thesis (in Greek) is included:
**KITSAKIS_G_IT219127_23.pdf**

Covers:
- Literature review
- Methodology
- Experiments and results
- Conclusions and future work

## Technologies Demonstrated

- **Deep Learning** - CNN, LSTM, GRU architectures
- **Time Series Analysis** - Sensor data processing
- **Model Optimization** - Hyperparameter tuning
- **Research Methodology** - Academic-level documentation
- **Python Programming** - Clean, modular code

## Citation

If you use this work, please cite:

```
Kitsakis, G. (2023). Smartphone Activity Recognition using Deep Learning.
Master's Thesis, Harokopio University of Athens.
```

## Contact

**Georgios Kitsakis**
- Email: kitsakisgk@gmail.com
- GitHub: @kitsakisGk
- LinkedIn: Georgios Kitsakis

---

**Master's Thesis Project - Demonstrating expertise in Deep Learning and Time Series Analysis**
