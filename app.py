from flask import Flask, render_template, request, jsonify
import torch
from app.cnn_classifier import CNNClassifier, Config
from app.pipeline import process_sequence
import os

app = Flask(__name__)

# Model configuration
config = Config()  # Using the Config class from cnn_classifier

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (1, 1, config.SEQUENCE_LENGTH)  # Input dimensions (batch, channel, sequence_length)
model = CNNClassifier(input_dim=input_dim, num_classes=1, config=config)
model_path = 'models/cnn_edna_classifier_20250929_015457.pth'

try:
    # Try loading with weights_only=False for backward compatibility
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    print("Model loaded successfully with weights_only=False")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please ensure the model file exists and is compatible")

model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the uploaded file temporarily
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    
    try:
        # Process the sequence and get prediction
        sequence = process_sequence(temp_path)
        with torch.no_grad():
            output = model(sequence)
            prediction = torch.sigmoid(output).item()
            
        result = {
            'prediction': prediction,
            'is_edna': prediction > 0.5,
            'confidence': prediction if prediction > 0.5 else 1 - prediction
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/database')
def database():
    return render_template('database.html')

if __name__ == '__main__':
    app.run(debug=True)