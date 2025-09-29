import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from app.cnn_classifier import CNNClassifier
from app.pipeline import load_and_preprocess_data, preprocess_sequence

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None, patience=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Model architecture:\n{model}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Debug info
                if batch_idx == 0:
                    print(f"\nInput shape: {inputs.shape}")
                    print(f"Labels shape: {labels.shape}")
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch: {epoch+1} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}')
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                
                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def plot_training_history(history, save_path=None):
    """Plot training history and optionally save the plot."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    data, labels = load_and_preprocess_data()
    
    # Create synthetic negative samples
    num_samples = len(data)
    synthetic_data = np.random.randint(65, 85, size=(num_samples, 1000))  # Random ASCII values for A-T-C-G
    synthetic_labels = np.zeros(num_samples)
    
    # Combine real and synthetic data
    data = np.vstack([data, synthetic_data])
    labels = np.concatenate([labels, synthetic_labels])
    
    # Split data into train and validation sets
    split_idx = int(0.8 * len(data))
    indices = np.random.permutation(len(data))
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, y_train = data[train_idx], labels[train_idx]
    X_val, y_val = data[val_idx], labels[val_idx]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution in training set: {np.bincount(y_train.astype(int))}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]  # Sequence length
    num_classes = len(torch.unique(y_train))
    model = CNNClassifier(input_dim=input_size, num_classes=num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Save model
    model_save_path = f'models/cnn_edna_classifier_{timestamp}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save training history
    history_save_path = f'results/training_history_{timestamp}.json'
    with open(history_save_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_save_path}")
    
    # Plot and save training history
    plot_save_path = f'results/training_summary_{timestamp}.png'
    plot_training_history(history, save_path=plot_save_path)
    print(f"Training plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()