import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate, Add, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
import time
from datetime import datetime, timedelta
import os
import cv2
from scipy.signal import find_peaks
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Set the size of the plots
plt.rcParams['figure.figsize'] = [9.375, 6]  # 750 pixels width (9.375 inches)
plt.style.use('ggplot')  # Use a more modern style for plots

# Connect to MetaTrader5 terminal
def connect_to_mt5():
    if not mt5.initialize():
        print("MetaTrader5 initialization failed")
        mt5.shutdown()
        return False
    print(f"MetaTrader5 successfully initialized. Version: {mt5.version()}")
    return True

# Get historical data
def get_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, num_bars=5000):
    """
    Get an extended set of historical data with additional technical indicators
    """
    now = datetime.now()
    from_date = now - timedelta(days=num_bars/24)  # Approximately for hourly bars

    print(f"Loading {symbol} data from {from_date} to {now}...")
    rates = mt5.copy_rates_range(symbol, timeframe, from_date, now)
    if rates is None or len(rates) == 0:
        print("Failed to load historical data")
        return None

    # Convert to pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.set_index('time', inplace=True)

    # Rename volume to tick_volume if necessary
    if 'volume' in rates_frame.columns and 'tick_volume' not in rates_frame.columns:
        rates_frame.rename(columns={'volume': 'tick_volume'}, inplace=True)

    # Add technical indicators
    rates_frame = add_technical_indicators(rates_frame)

    return rates_frame

def add_technical_indicators(df):
    """
    Add an extended set of technical indicators for improved CV model
    """
    # Check if tick_volume exists instead of volume
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df.rename(columns={'volume': 'tick_volume'}, inplace=True)

    # Simple Moving Averages
    df['sma7'] = df['close'].rolling(window=7).mean()
    df['sma25'] = df['close'].rolling(window=25).mean()
    df['sma99'] = df['close'].rolling(window=99).mean()

    # Exponential Moving Averages
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Stochastic Oscillator
    df['lowest_14'] = df['low'].rolling(window=14).min()
    df['highest_14'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - df['lowest_14']) / (df['highest_14'] - df['lowest_14']))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # ADX (Average Directional Index)
    df['tr'] = np.maximum(df['high'] - df['low'],
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                  abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) * 100

    # Rate of Change
    df['roc'] = ((df['close'] / df['close'].shift(12)) - 1) * 100

    # Derivative functions
    df['close_diff'] = df['close'].diff()
    df['close_diff_pct'] = df['close'].pct_change() * 100

    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()

    # Remove NaN values after calculating indicators
    df.dropna(inplace=True)

    return df

# Visualize all indicators for comprehensive data understanding
def plot_all_indicators(data, save_path='indicators_visualization.png'):
    """
    Creates a multi-panel plot with all major indicators for analysis
    """
    fig, axs = plt.subplots(5, 1, figsize=(9.375, 24), sharex=True)  # 750 pixels width (9.375 inches)

    # Price and SMA plot
    axs[0].plot(data.index, data['close'], label='Closing Price', color='black')
    axs[0].plot(data.index, data['sma7'], label='SMA 7', color='blue', alpha=0.7)
    axs[0].plot(data.index, data['sma25'], label='SMA 25', color='red', alpha=0.7)
    axs[0].plot(data.index, data['sma99'], label='SMA 99', color='green', alpha=0.7)
    axs[0].fill_between(data.index, data['bb_lower'], data['bb_upper'], color='gray', alpha=0.2, label='Bollinger Bands')
    axs[0].set_title('Price and Moving Averages')
    axs[0].legend()
    axs[0].grid(True)

    # MACD plot
    axs[1].plot(data.index, data['macd'], label='MACD', color='blue')
    axs[1].plot(data.index, data['macd_signal'], label='Signal Line', color='red')
    axs[1].bar(data.index, data['macd_hist'], label='Histogram', color='green', alpha=0.5)
    axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[1].set_title('MACD')
    axs[1].legend()
    axs[1].grid(True)

    # RSI plot
    axs[2].plot(data.index, data['rsi'], label='RSI', color='purple')
    axs[2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
    axs[2].axhline(y=30, color='green', linestyle='--', alpha=0.5)
    axs[2].set_title('RSI')
    axs[2].set_ylim(0, 100)
    axs[2].legend()
    axs[2].grid(True)

    # Stochastic Oscillator plot
    axs[3].plot(data.index, data['stoch_k'], label='%K', color='blue')
    axs[3].plot(data.index, data['stoch_d'], label='%D', color='red')
    axs[3].axhline(y=80, color='red', linestyle='--', alpha=0.5)
    axs[3].axhline(y=20, color='green', linestyle='--', alpha=0.5)
    axs[3].set_title('Stochastic Oscillator')
    axs[3].set_ylim(0, 100)
    axs[3].legend()
    axs[3].grid(True)

    # Volatility and ATR plot
    axs[4].plot(data.index, data['volatility'], label='Volatility (20)', color='orange')
    axs[4].plot(data.index, data['atr'], label='ATR (14)', color='brown')
    axs[4].set_title('Volatility')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Indicators plot saved as {save_path}")

# Create 2D images for computer vision
def create_2d_images(data, window_size=60, prediction_window=24, img_size=(128, 128)):
    """
    Create 2D images of all indicators for computer vision
    """
    images = []
    targets = []
    timestamps = []

    # List of indicators to include in the image
    indicators = ['close', 'open', 'high', 'low', 'tick_volume',
                 'sma7', 'sma25', 'sma99', 'ema12', 'ema26',
                 'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower',
                 'stoch_k', 'stoch_d', 'momentum', 'volatility']

    for i in range(len(data) - window_size - prediction_window):
        window_data = data.iloc[i:i+window_size]
        target_data = data.iloc[i+window_size:i+window_size+prediction_window]

        # Save timestamp for analysis
        timestamps.append(window_data.index[-1])

        # Create multi-channel image from indicators
        img_data = np.zeros((len(indicators), window_size))

        # Normalize each indicator separately
        for j, indicator in enumerate(indicators):
            # Use RobustScaler for robustness to outliers
            scaler = RobustScaler()
            if window_data[indicator].std() != 0:  # Avoid division by zero
                img_data[j] = scaler.fit_transform(window_data[indicator].values.reshape(-1, 1)).flatten()

        # For prediction, use both direction and magnitude of change
        price_direction = 1 if target_data['close'].iloc[-1] > window_data['close'].iloc[-1] else 0
        price_change_pct = (target_data['close'].iloc[-1] - window_data['close'].iloc[-1]) / window_data['close'].iloc[-1] * 100

        # Create a color RGB image for convolution
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)

        # Scale and transfer data to the image
        # Channel 1: Price indicators (red)
        for j in range(4):  # close, open, high, low
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 0] = 1.0  # Red channel for price data

        # Channel 2: Technical indicators (green)
        for j in range(4, 14):  # SMA, EMA, MACD, etc.
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 1] = 1.0  # Green channel for technical indicators

        # Channel 3: Oscillators (blue)
        for j in range(14, len(indicators)):  # RSI, stochastic, etc.
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 2] = 1.0  # Blue channel for oscillators

        # Add candlestick patterns
        # Find highs and lows
        for t in range(window_size):
            # Draw candle
            x = int(t / window_size * img_size[1])
            open_y = int((1 - img_data[1, t]) * img_size[0])  # open
            close_y = int((1 - img_data[0, t]) * img_size[0])  # close
            high_y = int((1 - img_data[2, t]) * img_size[0])  # high
            low_y = int((1 - img_data[3, t]) * img_size[0])   # low

            # Draw candle body
            if close_y < open_y:  # Bullish candle (green)
                for y in range(close_y, open_y):
                    if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                        img[y, x, 1] = 1.0  # Green
            else:  # Bearish candle (red)
                for y in range(open_y, close_y):
                    if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                        img[y, x, 0] = 1.0  # Red

            # Draw upper and lower wicks
            for y in range(high_y, min(close_y, open_y)):
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 0] = 0.5
                    img[y, x, 1] = 0.5
                    img[y, x, 2] = 0.5

            for y in range(max(close_y, open_y), low_y):
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 0] = 0.5
                    img[y, x, 1] = 0.5
                    img[y, x, 2] = 0.5

        # Blur for smoothing and highlighting general patterns
        img = gaussian_filter1d(img, sigma=0.5, axis=0)
        img = gaussian_filter1d(img, sigma=0.5, axis=1)

        images.append(img)
        targets.append([price_direction, price_change_pct])

    return np.array(images), np.array(targets), timestamps

# Save example images for visualization
def save_example_images(images, targets, timestamps, save_dir='example_images'):
    """
    Saves several example images for visual analysis
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save 5 random examples
    indices = np.random.choice(len(images), min(5, len(images)), replace=False)

    for i, idx in enumerate(indices):
        img = images[idx]
        target = targets[idx]
        timestamp = timestamps[idx]

        # Convert to image format (0-255)
        img_display = (img * 255).astype(np.uint8)

        # Add target information
        direction = "UP ▲" if target[0] > 0.5 else "DOWN ▼"
        change_pct = target[1]

        # Create caption on the image
        cv2.putText(img_display, f"Direction: {direction}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_display, f"Change: {change_pct:.2f}%", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_display, f"Date: {timestamp}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save image
        cv2.imwrite(f"{save_dir}/example_{i+1}.png", img_display)

    print(f"Example images saved in directory {save_dir}")

# Create an enhanced computer vision model with 2D-CNN for market analysis
def create_2d_cnn_model(input_shape):
    """
    Create a complex computer vision model for processing 2D images of market data
    with parallel paths for analyzing different aspects of the market
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # Modular architecture with separation into streams for different aspects of the market

    # 1. Stream for extracting local features (microstructures)
    local_path = Conv1D(64, 3, padding='same', activation='relu')(inputs[:,:,:,0])
    local_path = BatchNormalization()(local_path)
    local_path = Conv1D(64, 3, padding='same', activation='relu')(local_path)
    local_path = BatchNormalization()(local_path)
    local_path = MaxPooling1D(pool_size=2)(local_path)

    # 2. Stream for extracting trend features (macrostructures)
    trend_path = Conv1D(64, 7, padding='same', activation='relu')(inputs[:,:,:,1])
    trend_path = BatchNormalization()(trend_path)
    trend_path = Conv1D(64, 7, padding='same', activation='relu')(trend_path)
    trend_path = BatchNormalization()(trend_path)
    trend_path = MaxPooling1D(pool_size=2)(trend_path)

    # 3. Stream for extracting volatility and dynamics (oscillators)
    vol_path = Conv1D(64, 5, padding='same', activation='relu')(inputs[:,:,:,2])
    vol_path = BatchNormalization()(vol_path)
    vol_path = Conv1D(64, 5, padding='same', activation='relu')(vol_path)
    vol_path = BatchNormalization()(vol_path)
    vol_path = MaxPooling1D(pool_size=2)(vol_path)

    # Merge streams
    merged = Concatenate()([local_path, trend_path, vol_path])

    # Attention mechanism to highlight important time steps
    attention_layer = Attention()([merged, merged])
    attention_output = Add()([merged, attention_layer])

    # Bidirectional LSTM to account for long-term dependencies
    lstm = Bidirectional(LSTM(128, return_sequences=True))(attention_output)
    lstm = Dropout(0.3)(lstm)

    # Feature extraction
    pooled_features = GlobalAveragePooling1D()(lstm)

    # Fully connected layers for classification
    dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(pooled_features)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)

    dense = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)

    # Two outputs: price direction and percentage change
    output_direction = Dense(1, activation='sigmoid', name='direction')(dense)
    output_change = Dense(1, activation='linear', name='change_pct')(dense)

    # Create model
    model = Model(inputs=inputs, outputs=[output_direction, output_change])

    # Create model for feature extraction (for visualization)
    feature_model = Model(inputs=model.input, outputs=attention_layer)

    # Compile with different loss functions for different tasks
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'direction': 'binary_crossentropy',
            'change_pct': 'mean_squared_error'
        },
        metrics={
            'direction': 'accuracy',
            'change_pct': 'mae'
        },
        loss_weights={
            'direction': 1.0,
            'change_pct': 0.5
        }
    )

    return model, feature_model

# Train the model with an extended set of callbacks
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=5):
    """
    Train the model with an extended set of callbacks for improved training
    """
    import pathlib

    # Create directory for checkpoints
    try:
        checkpoint_dir = os.path.join(os.getcwd(), "model_checkpoints")
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"Successfully created directory for checkpoints: {checkpoint_dir}")
    except Exception as e:
        print(f"Error creating directory for checkpoints: {e}")
        checkpoint_dir = os.path.join(os.getcwd(), "model_checkpoints")
        print(f"Using default directory: {checkpoint_dir}")

    # Callbacks for improved training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_direction_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'  # Indicate that the accuracy metric should be maximized
        ),
        # Save the best model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_direction_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Dynamically adjust learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Separate target variables
    y_train_direction = y_train[:, 0]
    y_train_change = y_train[:, 1]
    y_val_direction = y_val[:, 0]
    y_val_change = y_val[:, 1]

    # Train the model
    history = model.fit(
        X_train,
        {'direction': y_train_direction, 'change_pct': y_train_change},
        validation_data=(
            X_val,
            {'direction': y_val_direction, 'change_pct': y_val_change}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history

# Extended analysis of model results
def evaluate_model(model, X_test, y_test):
    """
    Conducts a detailed evaluation of the model on the test set
    """
    # Separate target variables
    y_test_direction = y_test[:, 0]
    y_test_change = y_test[:, 1]

    # Prediction
    y_pred = model.predict(X_test)
    y_pred_direction = (y_pred[0] > 0.5).astype(int).flatten()
    y_pred_change = y_pred[1].flatten()

    # Calculate metrics for the price direction classification task
    accuracy = np.mean(y_pred_direction == y_test_direction)
    precision = precision_score(y_test_direction, y_pred_direction)
    recall = recall_score(y_test_direction, y_pred_direction)
    f1 = f1_score(y_test_direction, y_pred_direction)

    # Calculate metrics for the price change regression task
    mae = np.mean(np.abs(y_pred_change - y_test_change))
    mse = np.mean((y_pred_change - y_test_change)**2)

    # Generate report
    print("\n=== Model Evaluation ===")
    print(f"Price Direction Prediction Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nPrice Change MAE: {mae:.4f}%")
    print(f"Price Change MSE: {mse:.4f}%")

    # Plot confusion matrix
    cm = confusion_matrix(y_test_direction, y_pred_direction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Value')
    plt.xlabel('Predicted Value')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    # Plot predicted vs actual price changes
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_change, y_pred_change, alpha=0.5)
    plt.plot([-10, 10], [-10, 10], 'r--')
    plt.title('Predicted vs Actual Price Change')
    plt.xlabel('Actual Change, %')
    plt.ylabel('Predicted Change, %')
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig('price_change_prediction.png', dpi=150)
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae,
        'mse': mse
    }

# Visualize the training process
def plot_learning_history(history):
    """
    Extended visualization of the model training process
    """
    plt.figure(figsize=(12, 12))

    # Plot accuracy for the price direction classification task
    plt.subplot(2, 2, 1)
    plt.plot(history.history['direction_accuracy'])
    plt.plot(history.history['val_direction_accuracy'])
    plt.title('Price Direction Prediction Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot losses (use overall losses instead of direction_loss)
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot MAE for the price change regression task
    plt.subplot(2, 2, 3)
    plt.plot(history.history['change_pct_mae'])
    plt.plot(history.history['val_change_pct_mae'])
    plt.title('Price Change MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot losses for the price change regression task
    # Here we also need to change, as there is no separate key change_pct_loss
    plt.subplot(2, 2, 4)
    if 'change_pct_loss' in history.history:
        plt.plot(history.history['change_pct_loss'])
        plt.plot(history.history['val_change_pct_loss'])
    else:
        # If there are no separate losses, you can display something else or leave the plot empty
        plt.text(0.5, 0.5, 'Separate losses not available',
                 horizontalalignment='center', verticalalignment='center')

    plt.title('Losses for Price Change Task (if available)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_history.png', dpi=150)
    plt.close()
    print("Training history plot saved as 'learning_history.png'")

# Extended visualization of the model's attention maps
def visualize_attention_maps(feature_model, X_sample, y_sample, timestamps, save_path='attention_maps.png'):
    """
    Creates a visualization of the model's attention maps to understand
    which parts of the chart the model pays the most attention to
    """
    # Select several examples for visualization
    num_samples = min(3, len(X_sample))
    indices = np.random.choice(len(X_sample), num_samples, replace=False)

    # Get activations from the attention layer
    attention_maps = feature_model.predict(X_sample[indices])

    # Create figure
    fig, axs = plt.subplots(num_samples, 2, figsize=(14, 6*num_samples))

    for i in range(num_samples):
        idx = indices[i]
        img = X_sample[idx]
        direction = "UP ▲" if y_sample[idx, 0] > 0.5 else "DOWN ▼"
        change_pct = y_sample[idx, 1]
        timestamp = timestamps[idx]

        # Visualize the original image (price chart)
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f"Original Chart on {timestamp}\nPrediction: {direction}, Change: {change_pct:.2f}%")
        axs[i, 0].axis('off')

        # Visualize the attention map
        # Get the average attention value across channels and normalize
        attention = np.mean(np.abs(attention_maps[i]), axis=-1)
        attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention) + 1e-8)

        # Scale the attention map to the size of the image
        attention_resized = cv2.resize(attention, (img.shape[1], img.shape[0]))

        # Overlay the attention map on the original image
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Display the attention map
        axs[i, 1].imshow(heatmap)
        axs[i, 1].set_title("Model Attention Map")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Attention maps saved as '{save_path}'")

# Function for prediction with visualization
def make_prediction_with_visualization(model, feature_model, data, window_size=60, img_size=(128, 128), indicators=None):
    """
    Makes a prediction on the last available data window with full visualization
    """
    if indicators is None:
        indicators = ['close', 'open', 'high', 'low', 'tick_volume',
                     'sma7', 'sma25', 'sma99', 'ema12', 'ema26',
                     'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower',
                     'stoch_k', 'stoch_d', 'momentum', 'volatility']

    # Get the last data window
    last_window = data.iloc[-window_size:]

    # Create an image from the data
    img_data = np.zeros((len(indicators), window_size))

    # Normalize each indicator
    for j, indicator in enumerate(indicators):
        scaler = RobustScaler()
        if last_window[indicator].std() != 0:
            img_data[j] = scaler.fit_transform(last_window[indicator].values.reshape(-1, 1)).flatten()

    # Create a color RGB image
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)

    # Fill the image with data (similar to the create_2d_images function)
    # Channel 1: Price indicators (red)
    for j in range(4):  # close, open, high, low
        for t in range(window_size):
            x = int(t / window_size * img_size[1])
            y = int((1 - img_data[j, t]) * img_size[0])
            if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                img[y, x, 0] = 1.0

    # Channel 2: Technical indicators (green)
    for j in range(4, 14):
        for t in range(window_size):
            x = int(t / window_size * img_size[1])
            y = int((1 - img_data[j, t]) * img_size[0])
            if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                img[y, x, 1] = 1.0

    # Channel 3: Oscillators (blue)
    for j in range(14, len(indicators)):
        for t in range(window_size):
            x = int(t / window_size * img_size[1])
            y = int((1 - img_data[j, t]) * img_size[0])
            if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                img[y, x, 2] = 1.0

    # Add candlestick patterns
    for t in range(window_size):
        x = int(t / window_size * img_size[1])
        open_y = int((1 - img_data[1, t]) * img_size[0])
        close_y = int((1 - img_data[0, t]) * img_size[0])
        high_y = int((1 - img_data[2, t]) * img_size[0])
        low_y = int((1 - img_data[3, t]) * img_size[0])

        # Draw candle body
        if close_y < open_y:  # Bullish candle
            for y in range(close_y, open_y):
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 1] = 1.0
        else:  # Bearish candle
            for y in range(open_y, close_y):
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 0] = 1.0

        # Wicks
        for y in range(high_y, min(close_y, open_y)):
            if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                img[y, x, 0] = 0.5
                img[y, x, 1] = 0.5
                img[y, x, 2] = 0.5

        for y in range(max(close_y, open_y), low_y):
            if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                img[y, x, 0] = 0.5
                img[y, x, 1] = 0.5
                img[y, x, 2] = 0.5

    # Blur for smoothing
    img = gaussian_filter1d(img, sigma=0.5, axis=0)
    img = gaussian_filter1d(img, sigma=0.5, axis=1)

    # Transform for model input
    X_pred = np.expand_dims(img, axis=0)

    # Get prediction
    predictions = model.predict(X_pred)
    direction_prob = predictions[0][0][0]
    change_pct = predictions[1][0][0]

    # Interpret the result
    direction = "UP ▲" if direction_prob > 0.5 else "DOWN ▼"
    confidence = direction_prob if direction_prob > 0.5 else 1 - direction_prob

    # Get attention map
    attention_map = feature_model.predict(X_pred)[0]
    attention = np.mean(np.abs(attention_map), axis=-1)
    attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention) + 1e-8)

    # Create complex visualization
    fig = plt.figure(figsize=(15, 12))

    # 1. Price and indicators chart
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(last_window.index, last_window['close'], label='Closing Price', color='black', linewidth=2)
    ax1.plot(last_window.index, last_window['sma7'], label='SMA 7', color='blue', alpha=0.7)
    ax1.plot(last_window.index, last_window['sma25'], label='SMA 25', color='red', alpha=0.7)
    ax1.fill_between(last_window.index, last_window['bb_lower'], last_window['bb_upper'],
                     color='gray', alpha=0.2, label='Bollinger Bands')

    # Highlight areas with high attention
    high_attention_threshold = 0.7
    attention_resized = cv2.resize(attention, (window_size, 1))[0]  # Stretch to window length

    # Attention heatmap
    for i in range(len(last_window.index) - 1):
        ax1.axvspan(last_window.index[i], last_window.index[i+1],
                   alpha=attention_resized[i] * 0.3, color='red')

    # Points with high attention
    high_attention_indices = np.where(attention_resized > high_attention_threshold)[0]
    if len(high_attention_indices) > 0:
        high_attention_dates = [last_window.index[i] for i in high_attention_indices]
        high_attention_prices = [last_window['close'].iloc[i] for i in high_attention_indices]
        ax1.scatter(high_attention_dates, high_attention_prices, color='red', s=80, zorder=5, label='Attention Areas')

    ax1.set_title(f'Price Chart with Model Attention Areas')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Image created for computer vision
    ax2 = plt.subplot(3, 1, 2)
    ax2.imshow(img)
    ax2.set_title('2D Representation of Data for Computer Vision')
    ax2.axis('off')

    # 3. Prediction and attention map
    ax3 = plt.subplot(3, 1, 3)

    # Visualize prediction
    current_price = last_window['close'].iloc[-1]
    last_date = last_window.index[-1]

    # Create future dates for prediction
    future_dates = pd.date_range(start=last_date, periods=25, freq=last_window.index.to_series().diff().mode()[0])[1:]

    # Predicted change
    predicted_price = current_price * (1 + change_pct/100)

    # Display current price
    ax3.scatter(last_date, current_price, color='blue', s=100, zorder=5)
    ax3.annotate(f'Current Price: {current_price:.5f}',
                 xy=(last_date, current_price),
                 xytext=(10, -30),
                 textcoords='offset points',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Display prediction
    if direction == "UP ▲":
        arrow_color = 'green'
    else:
        arrow_color = 'red'

    ax3.annotate('', xy=(future_dates[-1], predicted_price),
                 xytext=(last_date, current_price),
                 arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color))

    ax3.annotate(f'Prediction: {direction} ({confidence*100:.1f}%)\nChange: {change_pct:.2f}%\nTarget: {predicted_price:.5f}',
                 xy=(future_dates[5], predicted_price),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    # Plot prediction range
    upper_bound = predicted_price * 1.005  # +0.5%
    lower_bound = predicted_price * 0.995  # -0.5%

    ax3.fill_between(future_dates, lower_bound, upper_bound, color=arrow_color, alpha=0.2)

    ax3.set_title(f'Price Prediction for the Next 24 Periods')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150)
    plt.close()
    print("Prediction visualization saved as 'prediction_visualization.png'")

    return {
        'direction': direction,
        'confidence': confidence * 100,
        'change_pct': change_pct,
        'current_price': current_price,
        'predicted_price': predicted_price
    }

# Create an animated GIF from a sequence of predictions
def create_prediction_animation(data, model, feature_model, window_size=60, prediction_steps=30, gif_path='prediction_animation.gif'):
    """
    Creates an animated GIF with a sequence of predictions based on a sliding window
    """
    import imageio

    # Create directory for temporary images
    temp_dir = 'temp_animation'
    os.makedirs(temp_dir, exist_ok=True)

    # Create a sequence of predictions
    frames = []

    for i in range(prediction_steps):
        # Select data window
        start_idx = len(data) - window_size - prediction_steps + i
        end_idx = start_idx + window_size
        window_data = data.iloc[start_idx:end_idx]

        # Make a prediction for this window
        # (simplified version of the make_prediction_with_visualization function)
        img_size = (128, 128)
        indicators = ['close', 'open', 'high', 'low', 'tick_volume',
                     'sma7', 'sma25', 'sma99', 'ema12', 'ema26',
                     'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower',
                     'stoch_k', 'stoch_d', 'momentum', 'volatility']

        # Create image
        img_data = np.zeros((len(indicators), window_size))
        for j, indicator in enumerate(indicators):
            scaler = RobustScaler()
            if window_data[indicator].std() != 0:
                img_data[j] = scaler.fit_transform(window_data[indicator].values.reshape(-1, 1)).flatten()

        # Create color RGB image
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)

        # Fill with data (shortened version)
        for j in range(4):  # Price data
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 0] = 1.0

        for j in range(4, 14):  # Technical indicators
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 1] = 1.0

        for j in range(14, len(indicators)):  # Oscillators
            for t in range(window_size):
                x = int(t / window_size * img_size[1])
                y = int((1 - img_data[j, t]) * img_size[0])
                if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                    img[y, x, 2] = 1.0

        # Blur
        img = gaussian_filter1d(img, sigma=0.5, axis=0)
        img = gaussian_filter1d(img, sigma=0.5, axis=1)

        # Get prediction
        X_pred = np.expand_dims(img, axis=0)
        predictions = model.predict(X_pred, verbose=0)
        direction_prob = predictions[0][0][0]
        change_pct = predictions[1][0][0]
        direction = "UP ▲" if direction_prob > 0.5 else "DOWN ▼"

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Price chart
        plt.subplot(2, 1, 1)
        plt.plot(window_data.index, window_data['close'], color='black', linewidth=2)
        plt.scatter(window_data.index[-1], window_data['close'].iloc[-1], color='blue', s=100)

        # Prediction
        current_price = window_data['close'].iloc[-1]
        predicted_price = current_price * (1 + change_pct/100)

        if direction == "UP ▲":
            arrow_color = 'green'
        else:
            arrow_color = 'red'

        # Prediction arrow
        last_date = window_data.index[-1]
        future_date = last_date + (window_data.index[1] - window_data.index[0]) * 5
        plt.annotate('', xy=(future_date, predicted_price),
                    xytext=(last_date, current_price),
                    arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color))

        plt.title(f'Frame {i+1}/{prediction_steps}: {window_data.index[-1].date()}\nPrediction: {direction}, Change: {change_pct:.2f}%')
        plt.grid(True, alpha=0.3)

        # Image for CV
        plt.subplot(2, 1, 2)
        plt.imshow(img)
        plt.title('Data Representation for Computer Vision')
        plt.axis('off')

        plt.tight_layout()

        # Save frame
        frame_path = f'{temp_dir}/frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=100)
        plt.close()

        frames.append(imageio.imread(frame_path))

    # Create GIF
    imageio.mimsave(gif_path, frames, duration=0.5)

    # Delete temporary files
    for i in range(prediction_steps):
        os.remove(f'{temp_dir}/frame_{i:03d}.png')
    os.rmdir(temp_dir)

    print(f"Prediction animation saved as '{gif_path}'")

# Main function
def main():
    print("Starting EURUSD prediction system using computer vision")

    # Connect to MT5
    if not connect_to_mt5():
        return

    print("Successfully connected to MetaTrader5")

    # Load historical data
    bars_to_load = 10000  # Increased sample size for better training
    data = get_historical_data(num_bars=bars_to_load)
    if data is None:
        mt5.shutdown()
        return

    print(f"Loaded {len(data)} bars of EURUSD history")

    # Visualize all indicators
    plot_all_indicators(data)

    # Convert data to images for computer vision
    print("Converting data to image format for computer vision...")
    window_size = 60  # Increased window size for better pattern capture
    img_size = (128, 128)  # Larger image size for more detailed convolution

    images, targets, timestamps = create_2d_images(data, window_size=window_size, img_size=img_size)
    print(f"Created {len(images)} images for training")

    # Save example images
    save_example_images(images, targets, timestamps)

    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test, timestamps_train_val, timestamps_test = train_test_split(
        images, targets, timestamps, test_size=0.2, shuffle=False
    )

    X_train, X_val, y_train, y_val, timestamps_train, timestamps_val = train_test_split(
        X_train_val, y_train_val, timestamps_train_val, test_size=0.2, shuffle=True, random_state=42
    )

    print(f"Dataset sizes: Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Create and train the computer vision model
    print("Training the computer vision model...")
    input_shape = images[0].shape  # (height, width, channels)

    model, feature_model = create_2d_cnn_model(input_shape)
    print(model.summary())

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=5)

    # Debug: print keys in history.history
    print("Keys in history.history:", history.history.keys())

    # Visualize the training process
    plot_learning_history(history)

    # Evaluate the model on the test set
    metrics = evaluate_model(model, X_test, y_test)

    # Visualize the model's attention maps
    visualize_attention_maps(feature_model, X_test[:10], y_test[:10], timestamps_test[:10])

    # Make a prediction with visualization
    prediction_result = make_prediction_with_visualization(model, feature_model, data, window_size=window_size, img_size=img_size)

    # Create an animated sequence of predictions
    create_prediction_animation(data, model, feature_model, window_size=window_size, prediction_steps=20)

    # Display the final prediction
    print("\n=== Final Prediction ===")
    print(f"Direction: {prediction_result['direction']} (confidence: {prediction_result['confidence']:.2f}%)")
    print(f"Percentage Change: {prediction_result['change_pct']:.2f}%")
    print(f"Current Price: {prediction_result['current_price']:.5f}")
    print(f"Predicted Price: {prediction_result['predicted_price']:.5f}")

    # Disconnect from MT5
    mt5.shutdown()
    print("Work completed")

if __name__ == "__main__":
    main()
