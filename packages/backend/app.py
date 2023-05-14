from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np
import cv2
from keras.models import Model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import img_to_array
from modules.predict import make_prediction 
import io # A core Python module for working with streams (sequential data that can be read and/or written).

app = Flask(__name__)
cors = CORS(app, resources={r"/api/analyze": {"origins": "*"}})

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
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
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
    # IMPORTANT
    # 1> The Alpha channel represents pixel transparency and isn't typically used when working with models like MobileNet in case of image with 4 channels
    # Use the model to extract features from the image
    # 2>
    image = preprocess_image(image)

    # Resolution
    width, height = image.shape[1], image.shape[0]
    resolution = width * height

    # TODO: Brightness and Contrast
    # gray_image = None
    # if len(image.shape) == 3:  # Color image
    #     if image.shape[2] == 3:  # RGB or BGR image
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     else:  # Something is not right, return an error
    #         return jsonify({'error': 'Invalid number of color channels'}), 400
    # elif len(image.shape) == 2:  # Grayscale image
    #     gray_image = image
    # else:  # Something is not right, return an error
    #     return jsonify({'error': 'Invalid image shape'}), 400

    # brightness = np.mean(gray_image)
    # contrast = np.std(gray_image)

    # Color Balance
    # mean_b, mean_g, mean_r = np.mean(image, axis=(0, 1))
    # std_b, std_g, std_r = np.std(image, axis=(0, 1))

    # # Noise Level
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # noise = np.std(image - blurred)

    # # Sharpness
    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # sharpness = np.var(laplacian)

    # Preprocess the image and extract features
    # image = preprocess_image(image)
    # These values correspond to the features extracted from the image by the pre-trained CNN model.
    # While these raw values can be difficult to interpret directly, 
    # they can be used as input to a machine learning model to make predictions.
    features = model.predict(image)

    return jsonify({
        'resolution': resolution,
        'features': features.tolist(),
        # 'brightness': brightness,
        # 'contrast': contrast,
        # 'color_balance': {'mean': [mean_b, mean_g, mean_r], 'std': [std_b, std_g, std_r]},
        # 'noise': noise,
        # 'sharpness': sharpness
    }), 200


    # For now, just return the features as a list
    return jsonify(features.tolist()), 200

@app.route('/api/predict/heuristic', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def heuristicPredict():
    # check if an image was received
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = np.array(image)

    # Criteria 1: Resolution - let's say we want at least 300x300 pixels
    if image.shape[0] < 300 or image.shape[1] < 300:
        return jsonify({'acceptability': 'low', 'reason': 'Resolution below 300x300'}), 200

    # Criteria 2: Brightness - we'll use OpenCV's cvtColor function to convert the image to grayscale,
    # then calculate the mean pixel value. Let's say we want this to be between 10 and 1000.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    if brightness < 10 or brightness > 1000:
        return jsonify({'acceptability': 'low', 'reason': 'Brightness outside acceptable range'}), 200

    # Criteria 3: Contrast - we'll calculate the standard deviation of the grayscale pixel values.
    # Let's say we want this to be at least 50.
    contrast = np.std(gray_image)
    if contrast < 50:
        return jsonify({'acceptability': 'low', 'reason': 'Contrast too low'}), 200

    # If the image passes all these tests, return a high acceptability score
    return jsonify({'acceptability': 'high'}), 200

@app.route('/api/predict/ml', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def mlPredict():
    # Possible approaches:
    # 1> Manually create a labeled dataset: This would involve gathering a collection of 
    # album artwork and manually assigning labels based on our criteria.
    # This could be time-consuming but would give you full control over the labeling process.

    # 2> Crowdsourcing: We could consider using a crowdsourcing platform like Amazon Mechanical Turk to label your images.
    # You'd provide guidelines on what makes a "good" or "bad" album cover, and then the crowd workers would assign labels.
    
    # 3> Use an existing labeled dataset: There are many publicly available image datasets that might be suitable for these needs.
    # However, finding a pre-labeled dataset that matches our specific criteria may be challenging.
    # For now I'm using "ImageNet Large Scale Visual Recognition Challenge (ILSVRC)" (155GB size of data)!
    

    # Get image from request
    image = request.files['image'].read()

    result = make_prediction(image)

    # Return the result
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
