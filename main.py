# main.py

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
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

def preprocess_image(image_path):
    """
    Preprocess the image to match the input requirements of the model.
    Converts the image to grayscale, resizes to 28x28, and normalizes pixel values.
    """
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_digit(image_path):
    """
    Predict the digit drawn in the image.
    """
    image_array = preprocess_image(image_path)
    if image_array is None:
        return "Error in image processing"
    try:
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions)
        return predicted_digit
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

def create_drawing_canvas():
    """
    Create a Tkinter canvas for users to draw digits.
    Allows saving the drawing as an image for prediction.
    """
    def save_drawing():
        # Save the drawing to a PNG file
        filename = "digit.png"
        image.save(filename)
        print(f"Drawing saved as {filename}")
        # Predict the digit
        predicted_digit = predict_digit(filename)
        print(f"Predicted Digit: {predicted_digit}")
        # Display the prediction
        result_label.config(text=f"Predicted Digit: {predicted_digit}")
        # Show the image with prediction
        plt.imshow(ImageOps.invert(image), cmap='gray')
        plt.title(f"Predicted: {predicted_digit}")
        plt.axis('off')
        plt.show()

    def draw_on_canvas(event):
        # Draw on the canvas
        x, y = event.x, event.y
        draw.line([(x-2, y-2), (x+2, y+2)], fill="white", width=8)
        canvas.create_oval(x-4, y-4, x+4, y+4, fill="white", outline="white")

    def clear_canvas():
        # Clear the canvas
        canvas.delete("all")
        draw.rectangle((0, 0, 200, 200), fill="black")
        print("Canvas cleared!")

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Digit Notepad")

    # Create a canvas for drawing
    canvas = tk.Canvas(root, width=200, height=200, bg="black")
    canvas.pack()

    # Create an image to store the drawing
    image = Image.new("L", (200, 200), "black")
    draw = ImageDraw.Draw(image)

    # Bind mouse events to the canvas
    canvas.bind("<B1-Motion>", draw_on_canvas)

    # Add buttons to save and clear the canvas
    button_frame = tk.Frame(root)
    button_frame.pack()

    save_button = tk.Button(button_frame, text="Save & Predict", command=save_drawing)
    save_button.pack(side=tk.LEFT)

    clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
    clear_button.pack(side=tk.LEFT)

    # Label to display the prediction result
    global result_label
    result_label = tk.Label(root, text="", font=("Arial", 16))
    result_label.pack()

    # Run the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    create_drawing_canvas()
