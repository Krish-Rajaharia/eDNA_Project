# eDNA Classifier

A deep learning-based environmental DNA (eDNA) sequence classifier built with PyTorch and Flask.

## Features

- CNN-based sequence classification
- Web interface for easy sequence submission
- Real-time classification results
- Database browser for reference sequences
- Model performance visualization
- Support for multiple sequence formats (FASTA, FASTQ, plain text)

## Tech Stack

- **Backend**: Python, Flask
- **ML/DL**: PyTorch, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite, BLAST database

## Project Structure

```
eDNA_Project/
├── app/
│   ├── __init__.py
│   ├── cnn_classifier.py
│   ├── download_data.py
│   ├── models.py
│   └── pipeline.py
├── data/
│   ├── 16S_ribosomal_RNA.fasta
│   └── 18S_fungal_sequences.fasta
├── models/
│   └── cnn_edna_classifier_[timestamp].pth
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html
│   ├── performance.html
│   └── database.html
├── uploads/
└── app.py
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/eDNA_Project.git
cd eDNA_Project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required data files:
```bash
python -c "from app.download_data import DataDownloader; from app.cnn_classifier import Config; DataDownloader(Config()).download_database('16S_ribosomal_RNA')"
python -c "from app.download_data import DataDownloader; from app.cnn_classifier import Config; DataDownloader(Config()).download_database('18S_fungal_sequences')"
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Upload a DNA sequence file (FASTA, FASTQ, or plain text)
2. Click "Classify" to process the sequence
3. View classification results and confidence scores
4. Browse the reference database for similar sequences
5. Check model performance metrics

## Model Architecture

- CNN-based architecture optimized for DNA sequence classification
- K-mer based sequence encoding
- Multiple convolutional layers with different filter sizes
- Dropout for regularization
- Binary classification output (eDNA vs non-eDNA)

## License

[Your chosen license]

## Contributors

[Your name/team]