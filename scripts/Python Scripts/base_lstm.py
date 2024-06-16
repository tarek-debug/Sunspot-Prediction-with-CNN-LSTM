import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Function to create dataset for LSTM
def create_dataset(X, time_steps=1, future_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(X[i + time_steps:i + time_steps + future_steps])
    return np.array(Xs), np.array(ys)

# Hyperparameters
TIME_STEPS = 48
future_steps_list = [24]

batch_size = 12  # You can adjust the batch size here

# Initialize results list
all_results = []

# Loop through each year
for year in range(2011, 2018 + 1):
    file_path = f'/home/tarek/2015/df_{year}.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")
        continue

    df = pd.read_csv(file_path)

    # Convert to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Extract and preprocess the relevant columns
    df_filtered = df[['time', 'White_Spot_Percentage']].dropna()
    df_filtered.set_index('time', inplace=True)

    # Resample to hourly frequency and interpolate missing values
    df_resampled = df_filtered.resample('H').asfreq()
    df_interpolated = df_resampled.interpolate(method='spline', order=2)

    # Prepare data for LSTM model
    data = df_interpolated['White_Spot_Percentage'].values.reshape(-1, 1)

    # Custom normalization to [-1, 1]
    column_min = data.min()
    column_max = data.max()
    data_scaled = ((2 * (data - column_min)) / (column_max - column_min)) - 1

    # Loop through each future step
    for FUTURE_STEPS in future_steps_list:
        X, y = create_dataset(data_scaled, TIME_STEPS, FUTURE_STEPS)
        y = y.reshape((y.shape[0], y.shape[1]))

        # Split data into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]

        # Build and compile the LSTM model
        model = Sequential([
            Bidirectional(LSTM(260, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
            Dropout(0.15),
            Bidirectional(LSTM(248)),
            Dropout(0.05),
            Dense(FUTURE_STEPS)  # Adjust the output layer to match FUTURE_STEPS
        ])

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, epochs=30, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Generate predictions
        y_pred = model.predict(X_val)

        # Calculate metrics for each future step
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)

        # Calculate Average White Spot and Average Normalized White Spot
        avg_white_spot = np.mean(df_interpolated['White_Spot_Percentage'])
        avg_normalized_white_spot = np.mean(data_scaled)

        # Check if early stopping was triggered
        early_stopped = early_stopping.stopped_epoch > 0

        all_results.append({
            'Year': year,
            'Future Steps': FUTURE_STEPS,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Average White Spot': avg_white_spot,
            'Average Normalized White Spot': avg_normalized_white_spot,
            'Early Stopped': early_stopped
        })

        # Plot predictions vs actual values for the first future step
        y_pred_first_step = y_pred[:, 0].reshape(-1, 1)
        y_val_first_step = y_val[:, 0].reshape(-1, 1)

        y_pred_original = (y_pred_first_step + 1) * (column_max - column_min) / 2 + column_min
        y_val_original = (y_val_first_step + 1) * (column_max - column_min) / 2 + column_min

        plt.figure(figsize=(12, 6))
        plt.plot(y_val_original, label='Actual', color='blue')
        plt.plot(y_pred_original, label='Predicted', color='red', linestyle='--')
        plt.title(f"LSTM Model Predictions vs Actual (First Future Step, {FUTURE_STEPS} Future Steps, Year {year})")
        plt.xlabel("Index")
        plt.ylabel("White_Spot_Percentage")
        plt.legend()

        # Save the plot as an image
        plt.savefig(f'/home/tarek/summer_2024/original_lstm/graphs/lstm_predictions_{year}_lookback{TIME_STEPS}_futuresteps{FUTURE_STEPS}_batch_size_12_epochs_30.png')
        plt.close()

# Save all results to a CSV file
results_df = pd.DataFrame(all_results)
results_df.to_csv('/home/tarek/summer_2024/original_lstm/lstm_predictions_lookback_48_futuresteps_24_batch_size_12_epochs_30.csv', index=False)

# Print the results for each future step
for result in all_results:
    print(f"Year: {result['Year']}")
    print(f"Future Steps: {result['Future Steps']}")
    print(f"MSE: {result['MSE']}")
    print(f"RMSE: {result['RMSE']}")
    print(f"MAE: {result['MAE']}")
    print(f"Average White Spot: {result['Average White Spot']}")
    print(f"Average Normalized White Spot: {result['Average Normalized White Spot']}")
    print(f"Early Stopped: {result['Early Stopped']}")
    print("\n")
