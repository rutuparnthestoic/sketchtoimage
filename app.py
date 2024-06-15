from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image

from test import load_model, preprocess_image, save_output

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'state_dict.pt' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, 'output_' + filename)
        file.save(input_path)
        
        input_image = preprocess_image(input_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_image = input_image.to(device)
        
        with torch.no_grad():
            output_image = model(input_image)
        
        save_output(output_image, output_path)
        
        return render_template('index.html', input_image=filename, output_image='output_' + filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
