import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from torchmetrics import MeanAbsoluteError
import time



# Huber loss
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        error = input - target
        is_small_error = torch.abs(error) < self.delta
        small_error_loss = 0.5 * torch.pow(error, 2)
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()
    
# Dataset class for loading and preprocessing image sequence data
class ImageSequenceDataset(Dataset):
    def __init__(self, csv_file, image_dir, time_steps=12, future_steps=12, wavelengths=['193', '171', '211', '304']):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.time_steps = time_steps
        self.wavelengths = wavelengths

        column_min = self.data['White_Spot_Percentage'].min()
        column_max = self.data['White_Spot_Percentage'].max()
        self.data['White_Spot_Percentage_Scaled'] = self.data['White_Spot_Percentage'].apply(
            lambda x: ((2 * (x - column_min)) / (column_max - column_min)) - 1)
        self.future_steps = future_steps

    def __len__(self):
        return len(self.data) - self.time_steps - self.future_steps + 2

    def __getitem__(self, idx):
        sequence_data = self.data.iloc[idx:idx + self.time_steps]
        sequence_images = []
        for _, row in sequence_data.iterrows():
            wavelength_images = []
            for wavelength in self.wavelengths:
                relative_path = row[wavelength]
                npz_file = np.load(os.path.join(self.image_dir, relative_path))
                image_array = npz_file['x']
                image_tensor = torch.from_numpy(image_array).float().clone()  # Use .float() and .clone() as needed
                wavelength_images.append(image_tensor)
            image_tensor = torch.cat(wavelength_images, dim=0)
            sequence_images.append(image_tensor)

        sequence_images = torch.stack(sequence_images)
        target_start_idx = idx + self.time_steps
        target = self.data.iloc[target_start_idx:target_start_idx+self.future_steps]['White_Spot_Percentage_Scaled']
        target = target.values.flatten()
        return sequence_images, torch.tensor(target, dtype=torch.float32)

    
class ResNetCNN(nn.Module):
    def __init__(self, input_channels=12, num_classes=64):
        super(ResNetCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Changed to ResNet50 for potentially better feature extraction
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    

class CNNLSTMModel(pl.LightningModule):
    def __init__(self, cnn_model, lstm_hidden_dim, future_steps, lr, num_features):
        super(CNNLSTMModel, self).__init__()
        self.lr = lr  # Ensure this is set to use it in configure_optimizers
        self.cnn_model = cnn_model
        self.lstm_hidden_dim = lstm_hidden_dim
        self.future_steps = future_steps
        self.num_features = num_features
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.dense = nn.Linear(lstm_hidden_dim * 2, future_steps)
        self.huber_loss = HuberLoss(delta=1.0)
        self.mae = MeanAbsoluteError()

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch_size, seq_len, C, H, W = x.size()
        cnn_embeddings = torch.zeros(batch_size, seq_len, self.num_features, device=x.device)
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            cnn_out = self.cnn_model(frame)
            cnn_embeddings[:, t, :] = cnn_out

        lstm_out, _ = self.lstm(cnn_embeddings)
        lstm_out = self.dropout(lstm_out)
        output = self.dense(lstm_out[:, -1, :])
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }
    def training_step(self, batch, batch_idx):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)  # Move to the correct device
            y_hat = self(x)
            loss = self.huber_loss(y_hat, y)
            mae = self.mae(y_hat, y)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  # Move to the correct device
        y_hat = self(x)
        val_loss = self.huber_loss(y_hat, y)
        val_mae = self.mae(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
        
    def calculate_final_mae(self, dataloader):
        self.eval()
        mae_metric = MeanAbsoluteError().to(self.device)
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self(X)
                mae_metric.update(y_hat, y)
        
        final_mae = mae_metric.compute()
        print(f"Final MAE: {final_mae.item()}")
        return final_mae.item()

    

def generate_and_plot_predictions(model, dataloader, save_path='model_predictions_plot.png'):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            y_hat = model(X)
            predictions.extend(y_hat.view(-1).cpu().numpy())
            actuals.extend(y.view(-1).cpu().numpy())

    # Assuming the predictions and actuals are already in the desired scale
    predictions_original = np.array(predictions).reshape(-1, 1)
    actuals_original = np.array(actuals).reshape(-1, 1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_original, label='Actual', color='blue')
    plt.plot(predictions_original, label='Predicted', color='red', linestyle='--')
    plt.title("Model Predictions vs Actual")
    plt.xlabel("Index")
    plt.ylabel("Target Variable")
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it in a notebook or GUI

    # Optionally, print the path to the saved plot
    print(f"Plot saved to {save_path}")


def calculate_rmse_numpy(predictions, actuals):
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    return rmse
def my_collate_fn(batch):
    sequences, targets = zip(*batch)
    
    # Determine the max target length in this batch
    max_target_len = max(target.size(0) for target in targets)

    # Filter and pad targets as needed
    filtered_sequences = []
    padded_targets = []
    for seq, target in zip(sequences, targets):
        if target.numel() > 0:  # Check if target is not empty
            filtered_sequences.append(seq)
            
            # Pad target if necessary
            if target.size(0) < max_target_len:
                # Use torch.nn.functional.pad or another method of your choice for padding
                padded_target = torch.nn.functional.pad(target, (0, max_target_len - target.size(0)), "constant", 0)
                padded_targets.append(padded_target)
            else:
                padded_targets.append(target)
    
    if not filtered_sequences:  # If all targets were empty, handle appropriately
        return None  # Adjust as needed
    
    sequences_batch = torch.stack(filtered_sequences)
    targets_batch = torch.stack(padded_targets)
    
    return sequences_batch, targets_batch



def main():
    start_time = time.time()  # Record the start time
    # Parameters
    batch_size = 10
    learning_rate = 1e-3
    num_epochs = 50
    model_save_path = '/home/tarek/2015/for_sheet/pretrained_models/model_48_24.pth'  # Define where to save the model

    # Load the dataset
    image_data_2015 = '/home/nasa_project/2015_data/SDO_MINI'
    dataset = ImageSequenceDataset(csv_file='/home/tarek/2015/df_2015_1_6.csv', image_dir=image_data_2015, time_steps=48, future_steps=24)

    # Initialize models with ResNetCNN
    cnn_model = ResNetCNN(input_channels=48, num_classes=128)
    combined_model = CNNLSTMModel(cnn_model=cnn_model, lstm_hidden_dim=128, future_steps=24, lr=learning_rate, num_features=128)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=my_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=my_collate_fn)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    # Set up PyTorch Lightning training with EarlyStopping with a patience of 10
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        accumulate_grad_batches=2,
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[early_stop_callback]
    )

    # Train the model
    trainer.fit(combined_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the entire model
    torch.save(combined_model, model_save_path)
    print(f'Model saved to {model_save_path}')
    
    # Set the model to evaluation mode and generate predictions
    combined_model.eval()
    #generate_and_plot_predictions(combined_model, val_loader, save_path='/home/tarek/2015/for_sheet/plot_2015_v8_12_6.png')

    # Calculate RMSE
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            y_hat = combined_model(X)
            predictions.extend(y_hat.view(-1).cpu().numpy())
            actuals.extend(y.view(-1).cpu().numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    rmse = calculate_rmse_numpy(predictions, actuals)
    print("RMSE:", rmse)
    # Calculate the average of the target variable
    average_target = dataset.data['White_Spot_Percentage'].mean()
    actuals_mean = np.full_like(actuals, fill_value=average_target)
    rmse_baseline = calculate_rmse_numpy(actuals_mean, actuals)
    print(f"Average target value: {average_target}")
    print(f"Baseline RMSE (predicting average): {rmse_baseline}")
    # Calculate min and max slope
    slopes = np.diff(predictions, axis=0).flatten()
    min_slope = slopes.min()
    max_slope = slopes.max()
    print(f"Min slope: {min_slope}")
    print(f"Max slope: {max_slope}")
    # After training, calculate the final MAE on the validation set
    final_mae = combined_model.calculate_final_mae(val_loader)
    print(f"The final MAE on the validation set is: {final_mae}")
    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total execution time
    print(f"Total execution time: {total_time:.2f} seconds")  # Print the total execution time

if __name__ == "__main__":
    main()

