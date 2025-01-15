import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Define the model path
MODEL_PATH = R'C:\Users\sd876\OneDrive\Desktop\DigitVision-Number-Predictor\models\digit_classifier_model.h5'

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure the model is trained and saved.")
    exit()

# Load and preprocess the MNIST test data
_, (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
test_images = test_images / 255.0

# Reshape the test images for CNN input
test_images = test_images.reshape((10000, 28, 28, 1))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test data
predictions = model.predict(test_images)

# Function to display a test image and its predicted label
def display_prediction(index):
    plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[index])
    true_label = test_labels[index]
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis('off')
    plt.show()

# Display predictions for the first 5 test images
for i in range(5):
    display_prediction(i)


