from Bio import SeqIO
import numpy as np
import torch
from pathlib import Path
import random
from typing import Tuple, List
import re


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess sequence data for training"""
    print("Loading sequence data...")
    
    data = []
    labels = []
    data_dir = Path("data")
    
    # Load positive examples (eDNA sequences)
    edna_files = list(data_dir.glob('16S_*.fasta'))
    edna_files.extend(data_dir.glob('18S_*.fasta'))
    
    for fasta_file in edna_files:
        sequences = load_fasta_sequences(fasta_file)
        data.extend(sequences)
        labels.extend([1] * len(sequences))  # 1 for eDNA
    
    # Load negative examples (synthetic sequences)
    synthetic_file = data_dir / 'synthetic_sequences.fasta'
    if synthetic_file.exists():
        sequences = load_fasta_sequences(synthetic_file)
        data.extend(sequences)
        labels.extend([0] * len(sequences))  # 0 for synthetic
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Shuffle the data
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    print(f"Loaded {len(data)} sequences ({sum(labels)} eDNA, {len(labels)-sum(labels)} synthetic)")
    return data, labels


def load_fasta_sequences(file_path: Path) -> List[np.ndarray]:
    """Load and preprocess sequences from a FASTA file"""
    sequences = []
    
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            # Convert sequence to numerical representation
            seq = preprocess_sequence(str(record.seq))
            if seq is not None:
                sequences.append(seq)
    
    return sequences


def preprocess_sequence(sequence: str) -> np.ndarray:
    """Preprocess a single DNA sequence"""
    # Remove non-ATCG characters
    sequence = re.sub('[^ATCG]', '', sequence.upper())
    
    # Skip if sequence is too short
    if len(sequence) < 100:
        return None
    
    # Convert to numerical representation
    seq_array = np.array([ord(c) for c in sequence])
    
    # Pad or truncate to fixed length
    if len(seq_array) > 1000:
        seq_array = seq_array[:1000]
    else:
        seq_array = np.pad(seq_array, (0, 1000 - len(seq_array)))
    
    return seq_array


def process_sequence(file_path):
    """Process a single sequence file and prepare it for classification"""
    sequences = []
    
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            # Convert sequence to numerical representation
            seq_array = np.array([ord(c) for c in str(record.seq)])
            # Pad or truncate to fixed length
            if len(seq_array) > 1000:
                seq_array = seq_array[:1000]
            else:
                seq_array = np.pad(seq_array, (0, 1000 - len(seq_array)))
            sequences.append(seq_array)
    
    if not sequences:
        raise ValueError("No valid sequences found in the file")
    
    # Convert to PyTorch tensor
    sequences = np.array(sequences)
    sequences = torch.FloatTensor(sequences).unsqueeze(0)  # Add batch dimension
    sequences = sequences.unsqueeze(1)  # Add channel dimension
    
    return sequences
