from flask import Flask, request, jsonify
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import io # A core Python module for working with streams (sequential data that can be read and/or written).

app = Flask(__name__)

# Load pre-trained model
base_model = MobileNet(weights='imagenet', include_top=True)

# Remove the last layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)


def preprocess_image(image):
    # Resize image to 224x224 pixels
    image = cv2.resize(image, (224, 224))

    # Convert image to array
    image = img_to_array(image)

    # Expand dimensions to fit model input
    image = np.expand_dims(image, axis=0)

    # Preprocess input for MobileNet
    image = preprocess_input(image)

    return image

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # check if an image was received
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400

    image = request.files['image'].read()
    # io.BytesIO is being used to treat a byte string as a file-like object, so that it can be passed to Image.open,
    # which expects a file or a file-like object.
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = np.array(image)

    # Preprocess the image
    image = preprocess_image(image)
    # 1> The Alpha channel represents pixel transparency and isn't typically used when working with models like MobileNet in case of image with 4 channels
    # Use the model to extract features from the image
    features = model.predict(image)

    # For now, just return the features as a list
    return jsonify(features.tolist()), 200

if __name__ == '__main__':
    app.run(debug=True)
