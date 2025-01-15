from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
MODEL_PATH = r"C:/Users/sd876/OneDrive/Desktop/DigitVision-Number-Predictor/models/digit_classifier_model.h5"

try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def preprocess_image(image):
    """
    Preprocess the image to match the input requirements of the model.
    Converts the image to grayscale, resizes to 28x28, and normalizes pixel values.
    """
    try:
        # Ensure that the image is in grayscale
        image = image.convert("L")
        # Resize to 28x28 pixels
        image = image.resize((28, 28))
        # Convert image to a numpy array and normalize pixel values
        image_array = np.array(image) / 255.0
        # Reshape image to include batch size and channel
        image_array = image_array.reshape(1, 28, 28, 1)
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_digit(image):
    """
    Predict the digit drawn in the image.
    """
    image_array = preprocess_image(image)
    if image_array is None:
        return "Error in image processing"
    try:
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions)
        return predicted_digit
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

# Test the prediction with a sample image
test_image_path = r"C:/Users/sd876/OneDrive/Desktop/DigitVision-Number-Predictor/data/sample_digit.png"  # Add a sample image path
try:
    test_image = Image.open(test_image_path)
    predicted_digit = predict_digit(test_image)
    print(f"Predicted digit: {predicted_digit}")
except Exception as e:
    print(f"Error opening or predicting image: {e}")
