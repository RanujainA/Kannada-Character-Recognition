from django.shortcuts import render
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('ipynb_file/cnn.h5')
# Create your views here.
def home_page(request):
    if request.method == 'POST' and request.FILES.get("file"):
        # Get the uploaded image
        img_file = request.FILES["file"]
        print(img_file)
        img = Image.open(img_file).convert("L")
        img = img.resize((64,64))

        # Convert image to NumPy array
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=[0, -1])  # Shape: (1, 64, 64, 1)
        prediction_prob = model.predict(img_array)
        prediction_label = np.argmax(prediction_prob, axis=1)[0]
        
        return render(request, 'home.html', {'output': prediction_label})
    else:
        return render(request, 'home.html')