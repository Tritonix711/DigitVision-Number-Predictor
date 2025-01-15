import os
import numpy as np
import gzip

def load_mnist_data(images_path, labels_path):
    """
    Load MNIST data from the specified paths for images and labels.
    Args:
        images_path (str): Path to the images file.
        labels_path (str): Path to the labels file.
    Returns:
        tuple: Tuple containing the images and corresponding labels as NumPy arrays.
    """
    # Load labels
    with open(labels_path, 'rb') as lbl_file:
        labels = np.frombuffer(lbl_file.read(), dtype=np.uint8, offset=8)

    # Load images
    with open(images_path, 'rb') as img_file:
        images = np.frombuffer(img_file.read(), dtype=np.uint8, offset=16)
        images = images.reshape(len(labels), 28, 28)

    return images, labels


def preprocess_data(images, labels):
    """
    Preprocess the images and labels.
    Args:
        images (np.ndarray): The image data.
        labels (np.ndarray): The label data.
    Returns:
        tuple: Preprocessed images and labels.
    """
    # Normalize images to range [0, 1]
    images = images.astype(np.float32) / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    labels_one_hot = np.zeros((labels.size, num_classes))
    labels_one_hot[np.arange(labels.size), labels] = 1

    return images, labels_one_hot


def load_and_preprocess_data(train_images_path, train_labels_path, test_images_path, test_labels_path):
    """
    Load and preprocess the MNIST data for training and testing.
    Args:
        train_images_path (str): Path to the training images file.
        train_labels_path (str): Path to the training labels file.
        test_images_path (str): Path to the testing images file.
        test_labels_path (str): Path to the testing labels file.
    Returns:
        tuple: Tuple containing training and testing images and labels.
    """
    # Load training data
    X_train, y_train = load_mnist_data(train_images_path, train_labels_path)

    # Load testing data
    X_test, y_test = load_mnist_data(test_images_path, test_labels_path)

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Define paths to dataset files
    base_path = "data/Custom CNN"
    train_images_path = os.path.join(base_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(base_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(base_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(base_path, "t10k-labels.idx1-ubyte")

    # Check if all files exist
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        train_images_path, train_labels_path, test_images_path, test_labels_path
    )

    # Display the shapes of the datasets
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Testing data: {X_test.shape}, {y_test.shape}")
