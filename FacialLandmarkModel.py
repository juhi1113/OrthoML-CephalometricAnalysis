import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
import os
import csv

class FacialLandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(os.path.join(self.data_dir, "dataset.csv"), "r") as csvfile:
            reader = csv.DictReader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                image_path = row["image_path"]
                #landmark_image_path = os.path.join(self.data_dir, "output")
                #landmark_image_filename = f"marked_{image_filename}"  # Example: "marked_p8SaniyaBarde.jpg"
                #landmark_image_path = os.path.join(self.data_dir, "output", landmark_image_filename)
                output_folder = os.path.join(self.data_dir, "output")
                for filename in os.listdir(output_folder):
                  if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    landmark_image_path = os.path.join(output_folder, filename)
                measurements = [float(value) for value in row["values"].split(",")]
            # Assign numerical labels to each measurement
                measurement_labels = {
                "SNA angle": 0,
                "SNB angle": 1,
                "ANB angle": 2,
                "Mandibular plane angle": 3,
                "Upper incisor to NA Linear": 4,
                "Upper incisor to NA angle": 5,
                "Lower incisor to NB Linear": 6,
                "Lower incisor to NB angle": 7,
                "Occlusal plane angle": 8,
                "Interincisal Angle": 9,
                "Nasolabial Angle": 10,
                "Nasomental Angle": 11
                         }
                measurements = [measurement_labels[measurement.strip()] for measurement in row["measurements"].split(",")]
                values = [float(value) for value in row["values"].split(",")]
                data.append((image_path, landmark_image_path, measurements, values))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      image_path, landmark_image_path, measurements, values = self.data[idx]

      image = cv2.imread(image_path)
      if image is None:
        raise FileNotFoundError(f"Failed to read image at path: {image_path}")

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if self.transform:
        image = self.transform(image)

      landmark_image = cv2.imread(landmark_image_path, cv2.IMREAD_GRAYSCALE)
      if landmark_image is None:
        # Handle read failure
        raise FileNotFoundError(f"Failed to read landmark image at path: {landmark_image_path}")

      if self.transform:
        landmark_image = self.transform(landmark_image)

      measurements = torch.tensor(measurements, dtype=torch.float32)
      values = torch.tensor(values, dtype=torch.float32)

      return image, landmark_image, measurements, values


class FacialLandmarkModel(nn.Module):
    def __init__(self, num_measurements):
        super(FacialLandmarkModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Replace the linear layer with an identity operation
        self.fc_measurements = nn.Linear(num_ftrs, num_measurements)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Reshape x to (batch_size, num_features)
        landmark_predictions = self.resnet.fc(x)  # No need for squeeze or permute
        measurements_predictions = self.fc_measurements(x)
        return landmark_predictions, measurements_predictions
    


data_dir = ""
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Repeat channels if grayscale
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FacialLandmarkDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get the number of measurements from the first data point
num_measurements = len(dataset.data[0][2])

# Train the model
model = FacialLandmarkModel(num_measurements)
landmark_criterion = nn.BCEWithLogitsLoss()
measurements_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30


for epoch in range(num_epochs):
    running_landmark_loss = 0.0
    running_measurements_loss = 0.0
    for images, landmark_images, measurements, values in train_loader:
        optimizer.zero_grad()
        landmark_predictions, measurements_predictions = model(images)
        landmark_images = landmark_images.view(landmark_images.size(0), -1).float() / 255.0
        #landmark_images = landmark_images.view(landmark_images.size(0), -1)
        landmark_images = landmark_images[:, :landmark_predictions.size(1)]
        landmark_loss = landmark_criterion(landmark_predictions, landmark_images.float())
        measurements_loss = measurements_criterion(measurements_predictions, values)
        total_loss = landmark_loss + measurements_loss
        total_loss.backward()
        optimizer.step()
        running_landmark_loss += landmark_loss.item()
        running_measurements_loss += measurements_loss.item()
    #print(f"Epoch {epoch + 1}, Training Landmark Loss: {running_landmark_loss / len(train_loader)}, Training Measurements Loss: {running_measurements_loss / len(train_loader)}")
    
    # Evaluate the model on the validation set
    model.eval()
    val_landmark_loss = 0.0
    val_measurements_loss = 0.0
    with torch.no_grad():
        for images, landmark_images, measurements, values in val_loader:
            landmark_predictions, measurements_predictions = model(images)
            landmark_images = landmark_images.view(landmark_images.size(0), -1).float() / 255.0
            landmark_images = landmark_images[:, :landmark_predictions.size(1)]
            landmark_loss = landmark_criterion(landmark_predictions, landmark_images.float())
            measurements_loss = measurements_criterion(measurements_predictions, values)
            val_landmark_loss += landmark_loss.item()
            val_measurements_loss += measurements_loss.item()
    #print(f"Epoch {epoch + 1}, Validation Landmark Loss: {val_landmark_loss / len(val_loader)}, Validation Measurements Loss: {val_measurements_loss / len(val_loader)}")




# Save the trained model
torch.save(model.state_dict(), "facial_landmark_model.pth")
