from django.shortcuts import render
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from Dataset.data import letters

def preprocess_uploaded_image(img_file):
    """Preprocess uploaded image: grayscale, resize, and remove noise."""
    img = Image.open(img_file).convert("L")  # Convert to grayscale
    img = np.array(img)
    img = cv2.resize(img, (64, 64)) 
    img = img / 255.0
    img = np.expand_dims(img, axis=[0, -1]) 
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
        prediction_label = np.argmax(prediction_prob, axis=1)[0]

        return render(request, 'home.html', {'output': letters[prediction_label+1]})
    else:
        return render(request, 'home.html')