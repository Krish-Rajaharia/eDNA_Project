from Bio import SeqIO
import numpy as np
import torch


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
