from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("alzymers_Model_64.h5")

# Ensure the 'uploads' folder exists for saving the uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Prediction content
alzheimer_info = {
    'Non Demented': {
        'explanation': "Brain is functioning normally with no signs of memory loss or confusion.",
        'medications': [
            "No Alzheimer’s medication needed.",
            "Vitamin B12 – supports brain and nerve function.",
            "Omega-3 – helps maintain brain health.",
            "Vitamin D – supports overall brain performance."
        ],
        'precautions': [
            "Get regular health checkups.",
            "Stay mentally and physically active.",
            "Eat a balanced diet.",
            "Avoid smoking and alcohol.",
            "Stay socially connected."
        ],
        'preventions': [
            "Daily exercise or walking.",
            "Memory games and reading.",
            "Proper sleep (7-8 hours).",
            "Control blood pressure and sugar levels.",
            "Limit stress and isolation."
        ]
    },
    'Very Mild Demented': {
        'explanation': "Minor memory issues that may seem like normal aging; person can still function independently.",
        'medications': [
            "Donepezil – supports brain communication.",
            "Rivastigmine – slows early memory decline.",
            "Medications help slow down progression, not cure it."
        ],
        'precautions': [
            "Maintain a daily routine.",
            "Use reminders and sticky notes.",
            "Avoid stressful situations.",
            "Get enough sleep.",
            "Stay engaged with friends or family."
        ],
        'preventions': [
            "Brain games like puzzles or Sudoku.",
            "Healthy meals rich in greens and fish.",
            "Physical activity like yoga or walking.",
            "Manage chronic health conditions.",
            "Stay socially and mentally active."
        ]
    },
    'Mild Demented': {
        'explanation': "Memory loss is noticeable; person may need help managing daily tasks like planning or finances.",
        'medications': [
            "Donepezil – boosts memory and awareness.",
            "Rivastigmine – helps manage symptoms.",
            "Galantamine – supports attention and mood."
        ],
        'precautions': [
            "Ensure a safe home environment.",
            "Use checklists and visual cues.",
            "Stick to routines.",
            "Caregiver support may be needed.",
            "Keep environment calm and familiar."
        ],
        'preventions': [
            "Take medicines regularly.",
            "Do familiar hobbies or crafts.",
            "Avoid multitasking.",
            "Daily walking or movement.",
            "Join community or support groups."
        ]
    },
    'Moderate Demented': {
        'explanation': "Serious confusion and memory loss; person may need help with basic tasks and safety.",
        'medications': [
            "Memantine – improves memory and reduces confusion.",
            "Used with Donepezil or Rivastigmine for better results.",
            "Helps manage mood swings and behavior."
        ],
        'precautions': [
            "Constant supervision may be needed.",
            "Use door locks to prevent wandering.",
            "Create a simple, calm routine.",
            "Monitor food and hygiene.",
            "Avoid loud noises or distractions."
        ],
        'preventions': [
            "Take medications without missing doses.",
            "Use calming music or visuals.",
            "Prevent falls (clear walking paths).",
            "Use simple instructions and familiar items.",
            "Encourage gentle, calming activities."
        ]
    }
}

# Prediction function
def predict_alzheimers(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    class_name = class_labels[class_idx]
    return class_name, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None
    medications = []
    precautions = []
    preventions = []
    img_path = None  # Variable to store uploaded image path

    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file.stream)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img.save(filename)  # Save the uploaded image
            label, confidence = predict_alzheimers(img)
            info = alzheimer_info.get(label, {})
            prediction = label
            explanation = info.get("explanation", "")
            medications = info.get("medications", [])
            precautions = info.get("precautions", [])
            preventions = info.get("preventions", [])
            img_path = url_for('static', filename='uploads/' + file.filename)  # URL of the uploaded image

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           explanation=explanation, medications=medications,
                           precautions=precautions, preventions=preventions, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)