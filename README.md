# LSTM Autoencoder for LDV Data Anomaly Detection

## Overview
This project implements an LSTM-based autoencoder for detecting anomalies in Laser Doppler Velocimetry (LDV) data. The model analyzes velocity measurements to identify unusual patterns using unsupervised learning techniques.

## Dependencies
```python
numpy
pandas
matplotlib
seaborn
sklearn
tensorflow
plotly
```

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn sklearn tensorflow plotly
```

## Data Structure
The project expects LDV data in the following format:
- Signal features stored in Excel files
- Features include: energy, std, maxabs, kurt, rms, fe, p2p, spectrms
- Two LDV channels (LDV1 and LDV2)

## Configuration
Set your dataset parameters at the start of the script:
```python
Dataset = 'KKK13'  # Dataset identifier
tstep = '0001'    # Time step parameter
```

## Data Preprocessing
1. Feature selection and cleaning
2. Log transformation for skewed features
3. Rolling window averaging (window=300)
4. Standard scaling
5. PCA-based dimensionality reduction

## Model Architecture
### Encoder
- Input Layer
- LSTM (128 units) + Dropout (0.2)
- LSTM (64 units) + Dropout (0.2)
- LSTM (16 units) + Dropout (0.2)
- LSTM (latent_dim)

### Decoder
- RepeatVector
- LSTM (16 units) + Dropout (0.2)
- LSTM (64 units) + Dropout (0.2)
- LSTM (128 units)
- TimeDistributed Dense Layer

## Usage

### Training
```python
# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")

# Train model
history = model.fit(X_train, X_train, 
                   epochs=50, 
                   batch_size=72, 
                   validation_split=0.1)
```

### Save/Load Model
```python
# Save model
save_path = './SDS_KKK13.h5'
model.save(save_path)

# Load model
model = keras.models.load_model(save_path)
```

### Anomaly Detection
```python
# Calculate reconstruction error threshold
threshold = np.max(train_mae_loss) * 2

# Detect anomalies
anomalies = test_mae_loss > threshold
```

## Visualization Features
1. Training/Validation Loss Curves
2. Latent Space Representation (2D/3D)
3. Anomaly Detection Results
4. ROC Curves for Performance Evaluation

## Model Variants
The repository includes multiple model architectures:

1. Standard Model (Default):
```python
def autoencoder_model(X, latent_dim):
    # Current implementation with 128->64->16->latent_dim architecture
```

2. Lightweight Model:
```python
def autoencoder_model(X):
    # Simplified architecture with 16->4 units
```

3. Deep Model:
```python
def autoencoder_model(X):
    # Enhanced architecture with additional layers
```

## Performance Metrics
- Mean Absolute Error (MAE) for reconstruction
- ROC-AUC score for anomaly detection
- Visual validation of detected anomalies

## Best Practices
1. Adjust the threshold multiplier based on your specific dataset
2. Use appropriate window size for rolling average (default=300)
3. Monitor validation loss to prevent overfitting
4. Adjust batch size based on your available memory

## Contributing
Feel free to submit issues and enhancement requests!


## citation
Yang, C., Kaynardag, K., Lee, G. W., & Salamone, S. (2025). Long short-term memory autoencoder for anomaly detection in rails using laser doppler vibrometer measurements. Journal of Nondestructive Evaluation, Diagnostics and Prognostics of Engineering Systems, 8(3), 031003.
