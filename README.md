![Screenshot 2025-01-15 102534](https://github.com/user-attachments/assets/a03fd1e6-d8ae-4840-8499-a15180e0517c)

# DigitVision - Number Predictor

**Digit Vision** is an interactive digit recognition application built with TensorFlow and Tkinter. The application allows users to draw digits on a canvas, and the model will predict the digit based on the drawing. This project uses a Convolutional Neural Network (CNN) for digit classification, leveraging the MNIST dataset for training.

## Features:
- **Digit Drawing Canvas**: Users can draw digits directly on the canvas using the mouse.
- **Digit Prediction**: Once a digit is drawn, the model predicts the digit and displays it.
- **Real-time Visualization**: The model's prediction is displayed in real-time after the drawing is saved.

---

## Project Structure:

```
├── data/                  # MNIST dataset (if applicable)
├── models/                # Saved models
├── notebooks/             # Optional experimentation and visualizations
├── src/                   # Core Python scripts
│   ├── __init__.py        # To make it a Python module
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── model_builder.py   # To define and compile the CNN model
│   ├── trainer.py         # For training and validating the model
│   ├── evaluator.py       # For evaluating the model
│   ├── predictor.py       # For making predictions with the model
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── main.py                # Main entry point for the Tkinter UI
```

---

## Installation

To get started, you'll need to set up the project environment and install the required dependencies.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/DigitVision-Number-Predictor.git
cd DigitVision-Number-Predictor
```

### Step 2: Set Up a Virtual Environment

Create a virtual environment to isolate your project dependencies:

```bash
python -m venv myenv
```

Activate the environment:
- **Windows**:
  ```bash
  myenv\Scripts\activate
  ```

- **Mac/Linux**:
  ```bash
  source myenv/bin/activate
  ```

### Step 3: Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Use

### Step 1: Load the Pre-trained Model

The model used for digit prediction is a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model is loaded from the `models/` directory in the project.

### Step 2: Drawing the Digit

- Launch the Tkinter-based GUI by running the `main.py` file. This will open a drawing canvas where you can draw digits.
  
- Draw a digit on the canvas using the mouse. The background color of the canvas is black, and the digits are drawn in white.

### Step 3: Saving the Drawing and Predicting

- Once you're done drawing, click the **Save & Predict** button. The drawing will be saved as `digit.png`.
  
- The image will be preprocessed (grayscale, resized to 28x28 pixels) and passed to the trained CNN model for digit prediction.

### Step 4: Viewing the Prediction

- After processing the image, the model will predict the digit you drew and display it in the application window.

- Additionally, the drawn image will be displayed with the predicted digit's title using `matplotlib`.

---

## Working Process

1. **Drawing the Digit**: 
   - A canvas is provided in the Tkinter window where the user can draw a digit. The canvas captures mouse movements to draw continuous lines.
   - When the user clicks "Save & Predict," the canvas is saved as an image (`digit.png`).

2. **Image Preprocessing**: 
   - The saved image is loaded, converted to grayscale, resized to 28x28 pixels, and normalized to match the input format expected by the CNN model.

3. **Prediction**:
   - The preprocessed image is passed through the CNN model to predict the digit.
   - The model’s output is processed to get the digit with the highest probability (the predicted digit).

4. **Displaying the Result**:
   - The predicted digit is displayed in the GUI, and the original drawing is shown with the prediction overlay using `matplotlib`.

---

## Model Details

- **Model Architecture**: 
   - The CNN model used in this project is based on the standard architecture used for MNIST digit classification tasks. It has convolutional layers followed by pooling layers, followed by fully connected layers at the end.
  
- **Training**:
   - The model was trained using the MNIST dataset, which contains handwritten digits from 0 to 9.
   - After training, the model is saved as an `.h5` file and is loaded during runtime to make predictions.

---

## Requirements

The project uses the following libraries:
- **TensorFlow**: For the CNN model and predictions.
- **PIL (Pillow)**: For image processing tasks.
- **Tkinter**: For the GUI and drawing canvas.
- **matplotlib**: For displaying the drawn image along with the prediction.

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Running the Application

To run the application, execute the `main.py` script:

```bash
python main.py
```

This will launch the Tkinter window where you can start drawing digits and get predictions.

---

## Troubleshooting

- **Model Not Loading**:
  - Ensure that the model file (`digit_classifier_model.h5`) exists in the `models/` directory.
  
- **Image Not Processing**:
  - Verify that the image is in the correct format and resolution. It should be 28x28 pixels for prediction.

---


## Contributing

### How to Contribute

1. **Fork the Repository**:
   - First, fork the project to your own GitHub account.
   - Navigate to the [GitHub repository](https://github.com/your-username/DigitVision-Number-Predictor) and click the "Fork" button in the top-right corner.

2. **Clone Your Fork**:
   - Clone your forked repository to your local machine:
     ```bash
     git clone https://github.com/your-username/DigitVision-Number-Predictor.git
     ```

3. **Create a New Branch**:
   - Before making changes, create a new branch to work on:
     ```bash
     git checkout -b feature-branch
     ```

4. **Make Your Changes**:
   - Make the necessary changes or enhancements to the project.
   - This could involve:
     - Writing new features.
     - Fixing bugs or issues.
     - Improving documentation.
   
5. **Commit Your Changes**:
   - After making your changes, commit them:
     ```bash
     git add .
     git commit -m "Add your commit message here"
     ```

6. **Push Your Changes**:
   - Push your changes to your forked repository:
     ```bash
     git push origin feature-branch
     ```

7. **Submit a Pull Request (PR)**:
   - Open a pull request from your forked repository's branch to the main repository's `main` branch.
   - Provide a description of your changes in the PR. If it's a bug fix, reference the issue number (e.g., "Fixes #25").

### Code of Conduct

By participating in this project, you agree to abide by our code of conduct. Please treat everyone with respect and kindness. We aim to maintain a welcoming and inclusive environment for all.

### Reporting Issues

If you find any bugs or have suggestions for improvements, feel free to open an issue by clicking on the "Issues" tab in this repository.

1. **Describe the Issue**: Clearly describe the bug or enhancement, and provide steps to reproduce it if applicable.
2. **Provide Environment Details**: Include information about the environment (e.g., Python version, OS).
3. **Include Error Messages**: If relevant, provide error logs or screenshots to help us understand the problem.

### Review Process

- Once your pull request is submitted, we will review it and may ask for further modifications.
- Contributions will be merged once they meet the project's standards and pass the necessary checks.

---

## Thank You!

We appreciate all contributions, whether big or small. Thank you for helping us improve **Digit Vision - Number Predictor**!

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to ask if you have any questions or need further assistance in setting up or using the project!

