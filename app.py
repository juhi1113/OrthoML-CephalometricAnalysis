import os
from flask import Flask, render_template, request
import torch
from torchvision import transforms
import cv2
import numpy as np
from FacialLandmarkModel import FacialLandmarkModel

app = Flask(__name__)

# Load the trained model
num_measurements = 12  # Replace with the actual number of measurements
model = FacialLandmarkModel(num_measurements=num_measurements)
model.load_state_dict(torch.load("facial_landmark_model.pth"))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Repeat channels if grayscale
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Measurement labels
measurement_labels = {
    0: "SNA angle",
    1: "SNB angle",
    2: "ANB angle",
    3: "Mandibular plane angle",
    4: "Upper incisor to NA Linear",
    5: "Upper incisor to NA angle",
    6: "Lower incisor to NB Linear",
    7: "Lower incisor to NB angle",
    8: "Occlusal plane angle",
    9: "Interincisal Angle",
    10: "Nasolabial Angle",
    11: "Nasomental Angle"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']

        # Read the image file
        image_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        # Run the model inference
        with torch.no_grad():
            _, measurements_predictions = model(image)

        # Process the measurement predictions
        measurements = measurements_predictions.squeeze().tolist()
        measurement_values = {measurement_labels[i]: value for i, value in enumerate(measurements)}

        return render_template('index.html', measurement_values=measurement_values)

    return render_template('index.html')

#if __name__ == '__main__':
#    app.run(debug=True)
