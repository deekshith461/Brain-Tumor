import os
from flask import Flask, request, render_template, redirect, url_for
from predict import model_predict

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # This loads the intro/home page with background and tumor info
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded image to static/uploads
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image_path = file_path

            # Make prediction
            prediction = model_predict(file_path)

    return render_template('predict.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
