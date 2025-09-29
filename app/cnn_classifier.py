#!/usr/bin/env python3
"""
CNN eDNA Classifier Training Script
Optimized for 500MB space constraint with NCBI BLAST database
"""

import os
import sys
import time
import urllib.request
import gzip
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import pickle
import json
from datetime import datetime

# Core ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install torch scikit-learn matplotlib seaborn")
    sys.exit(1)

class Config:
    """Configuration for CNN training"""
    # Data settings
    BASE_URL = "https://ftp.ncbi.nlm.nih.gov/blast/db/"
    MAX_SPACE_MB = 500
    SEQUENCE_LENGTH = 150  # Fixed length for CNN input
    KMER_SIZE = 6
    
    # Model settings optimized for 16S rRNA
    NUM_FILTERS = [64, 128, 256, 512]
    FILTER_SIZES = [3, 5, 7, 9]
    DROPOUT_RATE = 0.4
    EMBEDDING_DIM = 128
    
    # Training settings
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    
    # Paths
    DATA_DIR = Path("data")
    MODEL_DIR = Path("models")
    RESULTS_DIR = Path("results")

class DataDownloader:
    """Smart downloader for NCBI data with space constraints"""
    
    def __init__(self, config):
        self.config = config
        self.config.DATA_DIR.mkdir(exist_ok=True)
        
    def get_available_databases(self):
        """Get list of available BLAST databases"""
        print("🔍 Checking available BLAST databases...")
        
        # Small, taxonomy-rich databases that fit in 500MB
        priority_dbs = [
            "16S_ribosomal_RNA",  # ~50MB, great for microbial classification
            "18S_fungal_sequences",  # ~20MB, fungal sequences
            "28S_ribosomal_RNA",  # ~30MB, eukaryotic classification
            "ITS_RefSeq_Fungi",   # ~25MB, fungal ITS sequences
        ]
        
        return priority_dbs
        
    def download_database(self, db_name):
        """Download and extract specific BLAST database"""
        print(f"📥 Downloading {db_name}...")
        
        # Download database files
        db_files = ["nhr", "nin", "nsq"]  # Required BLAST db files
        success = True
        
        for ext in db_files:
            url = f"{self.config.BASE_URL}/{db_name}.{ext}"
            local_path = self.config.DATA_DIR / f"{db_name}.{ext}"
            
            try:
                if not local_path.exists():
                    print(f"Downloading {url}...")
                    urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")
                success = False
                break
        
        if not success:
            return False
            
        # Convert BLAST db to FASTA
        print(f"� Converting {db_name} to FASTA format...")
        output_fasta = self.config.DATA_DIR / f"{db_name}.fasta"
        
        try:
            # Install BLAST+ tools if not already installed
            if sys.platform == "win32":
                blast_cmd = "blastn"
            else:
                blast_cmd = "blastdbcmd"
            
            cmd = f"{blast_cmd} -db {self.config.DATA_DIR / db_name} -entry all -outfmt '%f' > {output_fasta}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"✅ Successfully created FASTA file: {output_fasta}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to convert to FASTA: {e}")
            print("💡 Installing BLAST+ tools might help. Try:")
            if sys.platform == "win32":
                print("Download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/")
            else:
                print("sudo apt-get install ncbi-blast+")
            return False
            
    def get_best_database(self):
        """Download the best database that fits in space constraint"""
        available_dbs = self.get_available_databases()
        
        for db_name in available_dbs:
            print(f"🎯 Trying to download {db_name}...")
            if self.download_database(db_name):
                return db_name
                
        # Fallback: create synthetic data
        print("⚠️ No database downloaded, creating synthetic dataset...")
        return self.create_synthetic_data()
        
    def create_synthetic_data(self):
        """Create synthetic DNA sequences for testing"""
        print("🧪 Creating synthetic eDNA dataset...")
        
        # Define taxonomic groups with characteristic sequence patterns
        taxa_patterns = {
            'bacteria': ['ATGC' * 10, 'GCTA' * 10, 'TACG' * 10],
            'archaea': ['CGAT' * 10, 'TAGC' * 10, 'ATCG' * 10],
            'fungi': ['GCAT' * 10, 'ACGT' * 10, 'TGCA' * 10],
            'protist': ['CATG' * 10, 'GTAC' * 10, 'AGTC' * 10],
            'plant': ['TCAG' * 10, 'GACT' * 10, 'CAGT' * 10]
        }
        
        sequences = []
        labels = []
        
        for taxon, patterns in taxa_patterns.items():
            for pattern in patterns:
                # Generate variations
                for i in range(200):  # 200 per pattern
                    seq = self._mutate_sequence(pattern, mutation_rate=0.1)
                    sequences.append(seq)
                    labels.append(taxon)
        
        # Save synthetic data
        synthetic_file = self.config.DATA_DIR / "synthetic_sequences.fasta"
        with open(synthetic_file, 'w') as f:
            for i, (seq, label) in enumerate(zip(sequences, labels)):
                f.write(f">{label}_{i}\n{seq}\n")
                
        print(f"✅ Created {len(sequences)} synthetic sequences")
        return "synthetic_sequences"
        
    def _mutate_sequence(self, sequence, mutation_rate=0.1):
        """Add random mutations to sequence"""
        bases = list(sequence)
        n_mutations = int(len(bases) * mutation_rate)
        
        for _ in range(n_mutations):
            pos = np.random.randint(len(bases))
            bases[pos] = np.random.choice(['A', 'T', 'C', 'G'])
            
        return ''.join(bases)

class DNAEncoder:
    """Encode DNA sequences for CNN input"""
    
    def __init__(self, sequence_length=150, encoding_type='onehot'):
        self.sequence_length = sequence_length
        self.encoding_type = encoding_type
        
        # Base mappings
        self.base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        self.int_to_base = {v: k for k, v in self.base_to_int.items()}
        
    def encode_sequence(self, sequence):
        """Encode single DNA sequence"""
        # Pad or truncate to fixed length
        sequence = sequence.upper()[:self.sequence_length]
        sequence = sequence.ljust(self.sequence_length, 'N')
        
        if self.encoding_type == 'onehot':
            return self._onehot_encode(sequence)
        elif self.encoding_type == 'integer':
            return self._integer_encode(sequence)
        elif self.encoding_type == 'kmer':
            return self._kmer_encode(sequence)
            
    def _onehot_encode(self, sequence):
        """One-hot encoding: (length, 5) for A,T,C,G,N"""
        encoding = np.zeros((self.sequence_length, 5))
        for i, base in enumerate(sequence):
            if base in self.base_to_int:
                encoding[i, self.base_to_int[base]] = 1
            else:
                encoding[i, 4] = 1  # Unknown base -> N
        return encoding
        
    def _integer_encode(self, sequence):
        """Integer encoding: (length,)"""
        return np.array([self.base_to_int.get(base, 4) for base in sequence])
        
    def _kmer_encode(self, sequence, k=6):
        """K-mer frequency encoding"""
        from collections import Counter
        
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmers.append(sequence[i:i+k])
            
        kmer_counts = Counter(kmers)
        
        # Convert to frequency vector (simplified)
        # In practice, you'd use all possible k-mers
        return np.array(list(kmer_counts.values())[:100])  # Top 100 k-mers

class DNADataset(Dataset):
    """PyTorch Dataset for DNA sequences"""
    
    def __init__(self, sequences, labels, encoder):
        self.sequences = sequences
        self.labels = labels
        self.encoder = encoder
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode sequence
        encoded_seq = self.encoder.encode_sequence(sequence)
        
        return torch.FloatTensor(encoded_seq), torch.LongTensor([label])

class CNNClassifier(nn.Module):
    """CNN for DNA sequence classification"""
    
    def __init__(self, input_dim, num_classes):
        super(CNNClassifier, self).__init__()
        
        # Model hyperparameters
        self.embedding_dim = 32
        self.conv_filters = [64, 128, 256]
        self.fc_sizes = [512, 128]
        self.dropout_rate = 0.5
        
        # First embedding layer to convert nucleotide values to embeddings
        self.embedding = nn.Linear(1, self.embedding_dim)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.embedding_dim
        for out_channels in self.conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            in_channels = out_channels
        
        # Calculate the size after convolutions
        conv_output_size = self.conv_filters[-1] * (input_dim // (2 ** len(self.conv_filters)))
        
        # Fully connected layers
        fc_layers = []
        in_features = conv_output_size
        for out_features in self.fc_sizes:
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convert input shape to (batch_size, sequence_length, 1)
        if len(x.shape) == 4:  # (batch, channel, height, width)
            x = x.squeeze(1)  # Remove channel dimension
        if len(x.shape) == 2:  # (batch, sequence_length)
            x = x.unsqueeze(-1)  # Add feature dimension
            
        # Apply embedding to each nucleotide
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Transpose for convolution layers (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

class SequenceProcessor:
    """Process FASTA files and extract sequences with labels"""
    
    def __init__(self, config):
        self.config = config
        
    def parse_fasta(self, fasta_file):
        """Parse FASTA file and extract sequences with taxonomic info"""
        sequences = []
        labels = []
        
        with open(fasta_file, 'r') as f:
            current_seq = ""
            current_label = ""
            
            for line in f:
                line = line.strip()
                
                if line.startswith('>'):
                    # Save previous sequence
                    if current_seq and current_label:
                        sequences.append(current_seq)
                        labels.append(current_label)
                    
                    # Extract taxonomic info from header
                    current_label = self._extract_taxonomy(line)
                    current_seq = ""
                else:
                    current_seq += line
            
            # Save last sequence
            if current_seq and current_label:
                sequences.append(current_seq)
                labels.append(current_label)
                
        return sequences, labels
        
    def _extract_taxonomy(self, header):
        """Extract taxonomic label from FASTA header"""
        # Simplified taxonomy extraction
        # In practice, you'd parse more complex taxonomic strings
        
        header = header.lower()
        
        if 'bacteria' in header or 'bacterial' in header:
            return 'bacteria'
        elif 'archaea' in header or 'archaeal' in header:
            return 'archaea'
        elif 'fungi' in header or 'fungal' in header:
            return 'fungi'
        elif 'plant' in header or 'viridiplantae' in header:
            return 'plant'
        elif 'protist' in header or 'protozoa' in header:
            return 'protist'
        else:
            return 'unknown'

def train_model(model, train_loader, val_loader, config, device):
    """Train CNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("🚀 Starting training...")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, val_accuracies

def check_gpu():
    """Check GPU availability and capabilities"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        
        print(f"🎮 GPU detected: {gpu_name}")
        print(f"📊 GPU Memory: {memory_allocated:.1f}MB / {memory_total:.1f}MB")
        
        # Test GPU performance
        print("🔥 Testing GPU performance...")
        x = torch.randn(1000, 1000).cuda()
        start_time = time.time()
        torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"⚡ GPU Matrix multiplication time: {gpu_time:.3f}s")
        
        return device
    else:
        print("💻 No GPU detected, using CPU")
        if torch.backends.mps.is_available():
            print("🍎 Apple M-series chip detected, using MPS backend")
            return torch.device('mps')
        return torch.device('cpu')

def main():
    """Main training pipeline"""
    config = Config()
    
    # Create directories
    for directory in [config.DATA_DIR, config.MODEL_DIR, config.RESULTS_DIR]:
        directory.mkdir(exist_ok=True)
    
    print("🧬 CNN eDNA Classifier Training Pipeline")
    
    # Check hardware and set device
    device = check_gpu()
    print(f"🖥️ Using device: {device}")
    
    # Look for available FASTA files
    fasta_files = list(config.DATA_DIR.glob("*.fasta"))
    if not fasta_files:
        print("❌ No FASTA files found in data directory!")
        print("💡 Please run download_data.py first to get the data.")
        return
    
    # Let user choose which file to use
    print("\nAvailable FASTA files:")
    for i, f in enumerate(fasta_files, 1):
        print(f"{i}. {f.name}")
    
    while True:
        try:
            choice = int(input("\nEnter file number to use for training: ")) - 1
            if 0 <= choice < len(fasta_files):
                fasta_file = fasta_files[choice]
                break
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Process sequences
    processor = SequenceProcessor(config)
    
    if not fasta_file.exists():
        print(f"❌ FASTA file not found: {fasta_file}")
        return
    
    print("📝 Processing sequences...")
    sequences, labels = processor.parse_fasta(fasta_file)
    
    print(f"✅ Processed {len(sequences)} sequences")
    print(f"📋 Taxonomic distribution: {Counter(labels)}")
    
    # Step 3: Encode sequences
    encoder = DNAEncoder(config.SEQUENCE_LENGTH, encoding_type='onehot')
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"🏷️  Number of classes: {num_classes}")
    print(f"🏷️  Classes: {label_encoder.classes_}")
    
    # Step 4: Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, encoded_labels, 
        test_size=config.VALIDATION_SPLIT, 
        stratify=encoded_labels, 
        random_state=42
    )
    
    # Step 5: Create datasets and loaders
    train_dataset = DNADataset(X_train, y_train, encoder)
    val_dataset = DNADataset(X_val, y_val, encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Step 6: Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Get input dimensions from first sample
    sample_input, _ = train_dataset[0]
    input_dim = sample_input.shape
    
    model = CNNClassifier(input_dim, num_classes, config).to(device)
    
    print(f"🧠 Model architecture:")
    print(model)
    
    # Step 7: Train model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, config, device
    )
    
    # Step 8: Save model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = config.MODEL_DIR / f"cnn_edna_classifier_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'label_encoder': label_encoder,
        'encoder': encoder,
        'input_dim': input_dim,
        'num_classes': num_classes
    }, model_path)
    
    print(f"💾 Model saved: {model_path}")
    
    # Save training history
    history_path = config.RESULTS_DIR / f"training_history_{timestamp}.json"
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'num_sequences': len(sequences),
        'num_classes': num_classes,
        'classes': label_encoder.classes_.tolist()
    }
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    class_counts = Counter(labels)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Taxonomic Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / f"training_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🎉 Training completed successfully!")
    print(f"📊 Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(f"📁 Results saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()