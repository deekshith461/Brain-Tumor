
# Brain Tumor Classification using VGG19 – Web Application

## 🧠 Project Overview

This project is a deep learning-based web application designed to classify brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. The model is built using the **VGG19** Convolutional Neural Network architecture and is integrated into a user-friendly **Flask** web application. Users can upload MRI images and instantly receive a prediction, making this project an excellent blend of machine learning and web development aimed at assisting early brain tumor d...

---

## 🖼️ Features

- Upload brain MRI images in JPG/JPEG/PNG format
- Classifies the image into one of the four tumor categories
- Uses pre-trained VGG19 architecture with fine-tuning
- Confusion matrix and evaluation metrics for model validation
- Flask-powered dynamic backend and HTML/CSS frontend
- Displays prediction with styling based on class (color-coded)

---

## 📁 Project Structure

```
brain_tumor_app_updated/
│
├── static/
│   ├── css/style.css              # Stylesheet
│   ├── images/                    # Background images (b1.png, b2.png)
│   └── uploads/                   # Stores uploaded images temporarily
│
├── templates/
│   ├── index.html                 # Home page
│   └── predict.html               # Upload & prediction interface
│
├── model/
│   └── multiclass_vgg19_model.h5 # Trained VGG19 model
│
├── tumordataset/                 # Dataset used for training/testing
│
├── app.py                        # Flask app backend
├── predict.py                    # Preprocessing & prediction logic
├── train_model.py                # Model training script
├── evaluate_model.py             # Evaluation & confusion matrix
├── confusion_matrix.png          # Output confusion matrix
└── requirements.txt              # Dependencies
```

---

## ⚙️ Technologies Used

- **Python 3.11**
- **TensorFlow / Keras**
- **OpenCV / NumPy**
- **Flask (Backend)**
- **HTML / CSS / JavaScript (Frontend)**
- **VGG19 Pre-trained Model (Transfer Learning)**

---

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain_tumor_classifier.git
   cd brain_tumor_classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open browser**
   ```
   Visit http://127.0.0.1:5000 in your browser
   ```

---

## 🧪 Dataset

The model was trained on a dataset of labeled brain MRI scans organized into 4 folders:

* `glioma`
* `meningioma`
* `pituitary`
* `notumor`

Each class contained images under `Training/` and `Testing/` directories. Preprocessing steps included resizing, filtering (Adaptive Bilateral Filter), and data augmentation.

---

## 🧠 Model: VGG19 Architecture

VGG19 (Visual Geometry Group) is a 19-layer deep CNN known for its simplicity and effectiveness. In this project:

* Pre-trained VGG19 was imported without the top classification layers.
* Custom dense layers with dropout were added for 4-class classification.
* The model was fine-tuned on the last 4 convolutional blocks.
* `softmax` activation was used for multiclass output.
* Achieved accuracy >96% on validation data.

---

## 🔗 Frontend–Backend Flow

1. **Frontend (HTML/CSS/JS)**: User uploads an image using the web form in `predict.html`.
2. **Backend (Flask)**: The image is received in `app.py` and passed to `predict.py`.
3. **Model (VGG19)**: `predict.py` loads and preprocesses the image, passes it to the model, and returns the predicted class.
4. **Response**: Flask renders the result with color-coded output (green for No Tumor, red for Tumor types).

---

## 📊 Model Evaluation

A confusion matrix was used to evaluate model performance. For example:

| Actual / Predicted | Glioma | Meningioma | No Tumor | Pituitary |
| ------------------ | ------ | ---------- | -------- | --------- |
| Glioma             | 287    | 12         | 0        | 1         |
| Meningioma         | 4      | 296        | 2        | 4         |
| No Tumor           | 0      | 2          | 403      | 0         |
| Pituitary          | 2      | 2          | 0        | 296       |

---

## 🎯 Objectives

* Build a 4-class brain tumor classifier with high accuracy
* Provide a web-based tool for real-time prediction
* Integrate VGG19 using Transfer Learning
* Deploy the app using Flask

---

## 🔮 Future Scope

* Add Grad-CAM visualizations for model explainability
* Deploy the app to cloud platforms like Heroku or AWS
* Use larger datasets and more complex architectures (like EfficientNet or ResNet)
* Enable mobile camera image input support

---

## 📄 License

This project is intended for educational and academic use only.
