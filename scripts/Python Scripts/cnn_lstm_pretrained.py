import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt
from torchmetrics import MeanAbsoluteError

from lstm_v11 import HuberLoss, ResNetCNN, ImageSequenceDataset, calculate_rmse_numpy, my_collate_fn

class CNNLSTMModel(pl.LightningModule):
    def __init__(self, cnn_model, lstm_hidden_dim, future_steps, lr, num_features):
        super(CNNLSTMModel, self).__init__()
        self.save_hyperparameters()
        self.cnn_model = cnn_model
        self.lstm_hidden_dim = lstm_hidden_dim
        self.future_steps = future_steps
        self.num_features = num_features
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.dense = nn.Linear(lstm_hidden_dim * 2, future_steps)
        self.huber_loss = HuberLoss(delta=1.0)
        self.mae = MeanAbsoluteError()

    def forward(self, x):
        if x.dim() == 4:  # Assuming x is [batch, channels, height, width]
            x = x.unsqueeze(1)  # Add a sequence dimension
        x = x.to(self.device)
        batch_size, seq_len, C, H, W = x.size()
        cnn_embeddings = torch.zeros(batch_size, seq_len, self.num_features, device=self.device)
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
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        loss = self.huber_loss(y_hat, y)
        mae = self.mae(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
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

def load_pretrained_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def retrain_model(model, train_loader, val_loader, max_epochs, device):
    model.train()
    trainer = pl.Trainer(max_epochs=max_epochs, logger=TensorBoardLogger("tb_logs", name="my_model_retraining"), callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model

def prepare_data_loaders(csv_file, image_dir, batch_size):
    dataset = ImageSequenceDataset(csv_file=csv_file, image_dir=image_dir, time_steps=48, future_steps=24)
    average_target = dataset.data['White_Spot_Percentage'].mean()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=my_collate_fn)
    return train_loader, val_loader, average_target

def generate_and_plot_predictions(model, dataloader, save_path):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X, y = X.to(model.device), y.to(model.device)
            y_hat = model(X)
            predictions.extend(y_hat.view(-1).cpu().numpy())
            actuals.extend(y.view(-1).cpu().numpy())
    predictions_original = np.array(predictions).reshape(-1, 1)
    actuals_original = np.array(actuals).reshape(-1, 1)
    plt.figure(figsize=(24, 12))
    plt.plot(actuals_original, label='Actual', color='blue')
    plt.plot(predictions_original, label='Predicted', color='red', linestyle='--')
    plt.title("Model Predictions vs Actual")
    plt.xlabel("Index")
    plt.ylabel("Target Variable")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    start_time = time.time()
    model_path = '/home/tarek/2015/for_sheet/pretrained_models/model_48_24.pth'
    model = load_pretrained_model(model_path, device)
    new_data_csv = '/home/tarek/2015/df_2012.csv'
    image_data_dir = '/home/stephen/data/SDO_MINI'
    batch_size = 12
    train_loader, val_loader, average_target = prepare_data_loaders(new_data_csv, image_data_dir, batch_size)
    learning_rate = 0.001
    max_epochs = 100
    model = retrain_model(model, train_loader, val_loader, max_epochs, device)
    torch.save(model, '/home/tarek/2015/for_sheet/pretrained_models/model_48_24.pth')
    print(f"Model saved to '/home/tarek/2015/for_sheet/pretrained_models/model_48_24.pth'")
    model.eval()
    generate_and_plot_predictions(model, val_loader, '/home/tarek/2015/for_sheet/plot_2012_v11_pretrained_48_24_output2.png')
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            predictions.extend(y_hat.view(-1).cpu().numpy())
            actuals.extend(y.view(-1).cpu().numpy())
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    rmse = calculate_rmse_numpy(predictions, actuals)
    print("RMSE:", rmse)
    final_mae = model.calculate_final_mae(val_loader)
    print(f"Final MAE: {final_mae}")
    actuals_mean = np.full_like(actuals, fill_value=average_target)
    rmse_baseline = calculate_rmse_numpy(actuals_mean, actuals)
    print(f"Baseline RMSE (predicting average): {rmse_baseline}")
    print(f"Average target value: {average_target}")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
