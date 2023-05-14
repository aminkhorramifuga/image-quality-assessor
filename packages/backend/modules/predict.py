from PIL import Image
import numpy as np
import io
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.basic_artwork_quality_CNN_model import BasicArtworkQualityCNNModel

# The goal is to create a binary label for each image indicating whether it's acceptable (1) or not (0). (A binary classification one.)
# Load trained model
model = BasicArtworkQualityCNNModel()
model.load_state_dict(torch.load("path/to/the/model.pth"))
model.eval()

# Define transformations
transform = Compose([
    Resize((224, 224)),  # Resize images to 224x224
    ToTensor(),  # Convert images to PyTorch tensors
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Load the dataset
dataset = ImageFolder(root="path/to/the/data", transform=transform)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def make_prediction(image_data):
    # Open image and convert to RGB
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Apply transformations
    image = transform(image)

    # Add an extra dimension (for the batch size)
    image = image.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Convert prediction to Python data types and return
    return {'prediction': predicted.item()}
