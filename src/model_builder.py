import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds and compiles a CNN model for digit classification.

    Args:
        input_shape (tuple): Shape of input images. Default is (28, 28, 1).
        num_classes (int): Number of output classes. Default is 10.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    
    # Pooling Layer 1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    
    # Pooling Layer 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten Layer
    model.add(Flatten())
    
    # Fully Connected Layer 1
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    cnn_model = create_model()
    cnn_model.summary()



