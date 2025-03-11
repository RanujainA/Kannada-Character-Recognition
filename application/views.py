from django.shortcuts import render
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def preprocess_uploaded_image(img_file):
    """Preprocess uploaded image: grayscale, resize, enhance contrast, and remove noise."""
    img = Image.open(img_file).convert("L")  # Convert to grayscale
    img = np.array(img)

    # Apply Adaptive Thresholding (convert to binary)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # Apply CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Apply Median Blur (Remove Noise)
    img = cv2.medianBlur(img, 3)

    # Resize and Normalize
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=[0, -1])  # Shape: (1, 64, 64, 1)
    
    return img

def home_page(request):
    model = load_model('ipynb_file/cnn.h5')
    if request.method == 'POST' and request.FILES.get("file"):
        # Get the uploaded image
        img_file = request.FILES["file"]
        
        # Preprocess image
        img_array = preprocess_uploaded_image(img_file)
        
        # Make prediction
        prediction_prob = model.predict(img_array)
        confidence = np.max(prediction_prob)  # Get highest probability
        prediction_label = np.argmax(prediction_prob, axis=1)[0]

        # If confidence is too low, return "Not sure"
        if confidence < 0.50:
            prediction_label = "Not sure"

        return render(request, 'home.html', {
            'output': prediction_label,
            'confidence': round(confidence, 2),
        })
    else:
        return render(request, 'home.html')