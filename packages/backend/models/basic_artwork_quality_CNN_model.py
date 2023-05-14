import torch.nn as nn
import torch.nn.functional as F

from offensive_text_detector import OffensiveTextDetector
from offensive_image_detector import OffensiveImageDetector

class BasicArtworkQualityCNNModel(nn.Module):
    def __init__(self):
        super(BasicArtworkQualityCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer: 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # Another convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer: 16*5*5 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)  # Another fully connected layer
        self.fc3 = nn.Linear(84, 2)  # Output layer: 84 input features, 2 output features

        # Initialize the text and image detectors
        self.text_detector = OffensiveTextDetector()
        self.image_detector = OffensiveImageDetector()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolutional layer, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolutional layer, ReLU activation, and pooling
        x = x.view(-1, 16 * 5 * 5)  # Reshape the tensor for the fully connected layers
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU activation
        x = F.relu(self.fc2(x))  # Apply second fully connected layer and ReLU activation
        x = self.fc3(x)  # Apply output layer

        # Add the results from the text and image detectors
        offensive_text = self.text_detector.detect(x)
        offensive_content = self.image_detector.detect(x)

        # Combine the results in some way # TODO: To be thought
        # For example, you could use a voting system where the image is considered unacceptable if two out of three detectors say it is.
        # This is just one possibility; 
        # we'll need to decide on the best approach based on your specific requirements and constraints.
        result = (x + offensive_text + offensive_content) / 3
        return result
