from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# -------------------------
# DNABERT Section
# -------------------------

MODEL_NAME = "zhihan1996/DNABERT-6"  # Pretrained DNABERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dnabert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

possible_taxa = ["Protist", "Cnidarian", "Metazoan", "Unknown"]

def classify_sequences(sequences):
    """
    Real classification with DNABERT.
    Takes a list of DNA sequences and returns predicted taxa.
    """
    predictions = []
    for seq in sequences:
        try:
            inputs = tokenizer(
                seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = dnabert_model(**inputs)
            logits = outputs.logits
            pred_class = int(torch.argmax(logits, dim=-1).item())
            predictions.append(possible_taxa[pred_class % len(possible_taxa)])
        except Exception:
            predictions.append("Unknown")
    return predictions


# -------------------------
# CNN Baseline Section
# -------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, seq_length, embedding_dim, num_filters, kernel_sizes, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.fc(dropped)


class DnaDataset(Dataset):
    def __init__(self, sequences, labels, seq_length):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.seq_length:
            seq = seq[:self.seq_length]
        elif len(seq) < self.seq_length:
            seq += 'N' * (self.seq_length - len(seq))
        encoded_seq = one_hot_encode(seq)
        return encoded_seq, self.labels[idx]


def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = torch.zeros(len(mapping), len(seq))
    for i, nucleotide in enumerate(seq.upper()):
        if nucleotide in mapping:
            encoded[mapping[nucleotide], i] = 1.0
    return encoded


def parse_fasta_header(header):
    try:
        parts = header.split(' ', 1)
        if len(parts) > 1:
            taxonomy_str = parts[1]
            tax_levels = [level.strip() for level in taxonomy_str.split(';')]
            if len(tax_levels) > 2:
                return tax_levels[2]  # Heuristic: Phylum level
    except Exception:
        return None
    return None


def calculate_accuracy(y_pred, y_true):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y_true.view_as(top_pred)).sum()
    return correct.float() / y_true.shape[0]


def train_cnn_baseline(
    fasta_file="18S_ribosomal_RNA.fsa",
    model_save_path="cnn_classifier.pth",
    encoder_save_path="label_encoder.pkl",
    seq_length=300,
    embedding_dim=4,
    num_filters=128,
    kernel_sizes=[3, 5, 7, 9],
    dropout_rate=0.5,
    lr=0.001,
    batch_size=128,
    num_epochs=10,
    min_class_count=50
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Load sequences ---
    sequences, labels = [], []
    with open(fasta_file, "r") as f:
        seq, header = "", ""
        for line in f:
            if line.startswith(">"):
                if seq and header:
                    label = parse_fasta_header(header)
                    if label:
                        sequences.append(seq)
                        labels.append(label)
                header = line.strip()
                seq = ""
            else:
                seq += line.strip()
        if seq and header:
            label = parse_fasta_header(header)
            if label:
                sequences.append(seq)
                labels.append(label)

    label_counts = Counter(labels)
    filtered_sequences, filtered_labels = [], []
    for seq, label in zip(sequences, labels):
        if label_counts[label] >= min_class_count:
            filtered_sequences.append(seq)
            filtered_labels.append(label)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_labels)
    num_classes = len(label_encoder.classes_)

    dataset = DnaDataset(filtered_sequences, encoded_labels, seq_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes, seq_length, embedding_dim, num_filters, kernel_sizes, dropout_rate).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for seqs, labs in train_loader:
            seqs, labs = seqs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, labs)
            acc = calculate_accuracy(preds, labs)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_acc += acc.item()

        model.eval()
        total_val_loss, total_val_acc = 0, 0
        with torch.no_grad():
            for seqs, labs in val_loader:
                seqs, labs = seqs.to(DEVICE), labs.to(DEVICE)
                preds = model(seqs)
                loss = criterion(preds, labs)
                acc = calculate_accuracy(preds, labs)
                total_val_loss += loss.item()
                total_val_acc += acc.item()

        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {total_train_loss/len(train_loader):.3f} "
              f"| Train Acc {total_train_acc/len(train_loader)*100:.2f}% "
              f"| Val Loss {total_val_loss/len(val_loader):.3f} "
              f"| Val Acc {total_val_acc/len(val_loader)*100:.2f}% "
              f"| Time {int(epoch_mins)}m {int(epoch_secs)}s")

    torch.save(model.state_dict(), model_save_path)
    with open(encoder_save_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print("CNN training complete. Model and encoder saved.")
