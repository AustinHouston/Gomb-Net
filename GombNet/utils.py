import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.init as init

def train_model(model, train_loader, val_loader, n_epochs, criterion, optimizer, device, save_name, save_loss_history = True, save_checkpoints = None):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    model.to(device)
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

            val_epoch_loss = val_running_loss / len(val_loader)
            val_loss_history.append(val_epoch_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

        if save_loss_history:
            np.savez(save_name + '_loss_history.npz', train_loss_history=train_loss_history, val_loss_history=val_loss_history)
        # Save model and optimizer state if validation loss improved
        if save_name:
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                checkpoint = {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss}
                torch.save(checkpoint, str(save_name+ '_best.pth'))
                print(f"Model saved as validation loss improved to {val_epoch_loss:.4f}")
        if save_checkpoints:
            if epoch in save_checkpoints:
                checkpoint = {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss}
                torch.save(checkpoint, str(save_name + f'_checkpoint_{epoch}.pth'))
                print(f"Model saved as checkpoint at epoch {epoch}")

    return model, train_loss_history, val_loss_history


def predict(model, image, device='mps'):
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained PyTorch model
        image: Input image as tensor (C, H, W) or numpy array
        device: Device to run on ('mps' or 'cpu')
    
    Returns:
        prediction: Output as numpy array (C, H, W)
    """
    model.eval()
    model.to(device)
    
    # Convert to tensor if numpy
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Add batch dimension if needed
    if image.ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
    
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)
        prediction = prediction.cpu().numpy()
    
    # Remove batch dimension
    if prediction.shape[0] == 1:
        prediction = prediction[0]  # (C, H, W)
    
    return prediction


def predict_batch(model, dataloader, device='mps', max_batches=None):
    """
    Make predictions on a full dataloader.
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images
        device: Device to run on 
        max_batches: Maximum number of batches to predict (None = all)
    Returns:
        predictions: List of prediction arrays
        targets: List of target arrays (if available)
        inputs: List of input arrays
    """
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    inputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            
            # Unpack batch
            if len(batch) == 2:
                imgs, tgts = batch
                imgs = imgs.to(device)
                tgts = tgts.to(device)
                
                # Predict
                preds = model(imgs)
                
                # Store results
                predictions.extend(preds.cpu().numpy())
                targets.extend(tgts.cpu().numpy())
                inputs.extend(imgs.cpu().numpy())
            else:
                imgs = batch[0].to(device)
                preds = model(imgs)
                predictions.extend(preds.cpu().numpy())
                inputs.extend(imgs.cpu().numpy())
    
    return predictions, targets, inputs