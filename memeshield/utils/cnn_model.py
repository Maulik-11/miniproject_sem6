"""
MemeShield CNN Feature Extractor Module
Uses pre-trained CNN models (MobileNetV2/ResNet50) for visual feature extraction.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class CNNFeatureExtractor:
    """
    CNN Feature Extractor using pre-trained models.
    Extracts visual features from meme images.
    """
    
    def __init__(self, model_name='MobileNetV2', input_shape=(224, 224, 3)):
        """
        Initialize the CNN feature extractor.
        
        Args:
            model_name (str): Name of the pre-trained model ('MobileNetV2' or 'ResNet50').
            input_shape (tuple): Input shape for images (height, width, channels).
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = None
        self.feature_extractor = None
        self.feature_dim = None
        
        # Initialize the model
        self._build_model()
    
    def _build_model(self):
        """Build the feature extraction model."""
        print(f"Loading {self.model_name} feature extractor...")
        
        if self.model_name == 'MobileNetV2':
            # Load MobileNetV2 with pre-trained ImageNet weights
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg'  # Global average pooling
            )
            self.preprocess_fn = mobilenet_preprocess
            self.feature_dim = 1280
            
        elif self.model_name == 'ResNet50':
            # Load ResNet50 with pre-trained ImageNet weights
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg'
            )
            self.preprocess_fn = resnet_preprocess
            self.feature_dim = 2048
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Use 'MobileNetV2' or 'ResNet50'.")
        
        # Freeze the base model weights
        base_model.trainable = False
        
        # Create feature extraction model
        self.feature_extractor = base_model
        self.model = base_model
        
        print(f"Feature extractor loaded. Output dimension: {self.feature_dim}")
    
    def preprocess(self, image_array):
        """
        Preprocess image for the specific CNN model.
        
        Args:
            image_array (numpy.ndarray): Image array of shape (batch, H, W, 3) 
                                        with values in [0, 1].
        
        Returns:
            numpy.ndarray: Preprocessed image array.
        """
        # Convert from [0, 1] to [0, 255] for the preprocess function
        image_scaled = image_array * 255.0
        
        # Apply model-specific preprocessing
        preprocessed = self.preprocess_fn(image_scaled)
        
        return preprocessed
    
    def extract_features(self, image_array):
        """
        Extract visual features from an image.
        
        Args:
            image_array (numpy.ndarray): Preprocessed image array of shape (1, H, W, 3).
        
        Returns:
            numpy.ndarray: Feature vector of shape (feature_dim,).
        """
        try:
            # Ensure correct input shape
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Preprocess for the specific model
            preprocessed = self.preprocess(image_array)
            
            # Extract features
            features = self.feature_extractor.predict(preprocessed, verbose=0)
            
            # Flatten to 1D if needed
            features = features.flatten()
            
            return features
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return zero vector as fallback
            return np.zeros(self.feature_dim)
    
    def extract_features_batch(self, image_batch):
        """
        Extract features from a batch of images.
        
        Args:
            image_batch (numpy.ndarray): Batch of images (N, H, W, 3).
        
        Returns:
            numpy.ndarray: Feature matrix of shape (N, feature_dim).
        """
        try:
            preprocessed = self.preprocess(image_batch)
            features = self.feature_extractor.predict(preprocessed, verbose=0)
            return features
        except Exception as e:
            print(f"Batch feature extraction error: {e}")
            return np.zeros((image_batch.shape[0], self.feature_dim))
    
    def get_feature_dim(self):
        """Get the dimension of the feature vector."""
        return self.feature_dim


class HateSpeechImageClassifier:
    """
    A CNN classifier specifically trained for hate speech detection in images.
    This is a skeleton for training a custom classifier.
    """
    
    def __init__(self, num_classes=2, model_path=None):
        """
        Initialize the hate speech image classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for hate/non-hate).
            model_path (str): Path to load a pre-trained model.
        """
        self.num_classes = num_classes
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build a custom CNN classifier on top of MobileNetV2."""
        # Base feature extractor
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build classifier on top
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Image classifier model built.")
    
    def train(self, train_images, train_labels, validation_data=None, epochs=10, batch_size=32):
        """
        Train the classifier.
        
        Args:
            train_images: Training images array.
            train_labels: Training labels (one-hot encoded).
            validation_data: Tuple of (val_images, val_labels).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        
        Returns:
            History object from training.
        """
        # Preprocess images
        train_images = mobilenet_preprocess(train_images * 255.0)
        
        if validation_data:
            val_images, val_labels = validation_data
            val_images = mobilenet_preprocess(val_images * 255.0)
            validation_data = (val_images, val_labels)
        
        # Train
        history = self.model.fit(
            train_images, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        return history
    
    def predict(self, image_array):
        """
        Predict hate speech probability for an image.
        
        Args:
            image_array: Image array of shape (1, H, W, 3).
        
        Returns:
            tuple: (predicted_class, confidence)
        """
        # Preprocess
        preprocessed = mobilenet_preprocess(image_array * 255.0)
        
        # Predict
        predictions = self.model.predict(preprocessed, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        label = 'hate' if predicted_class == 1 else 'non-hate'
        
        return label, float(confidence)
    
    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


# Utility functions

def extract_image_statistics(image_array):
    """
    Extract statistical features from an image.
    
    Args:
        image_array: Image array of shape (H, W, 3) with values in [0, 1].
    
    Returns:
        dict: Dictionary of image statistics.
    """
    # Remove batch dimension if present
    if len(image_array.shape) == 4:
        image_array = image_array[0]
    
    return {
        'mean_r': float(np.mean(image_array[:, :, 0])),
        'mean_g': float(np.mean(image_array[:, :, 1])),
        'mean_b': float(np.mean(image_array[:, :, 2])),
        'std_r': float(np.std(image_array[:, :, 0])),
        'std_g': float(np.std(image_array[:, :, 1])),
        'std_b': float(np.std(image_array[:, :, 2])),
        'brightness': float(np.mean(image_array)),
        'contrast': float(np.std(image_array))
    }


# Test function
if __name__ == "__main__":
    print("Testing CNN Feature Extractor...")
    
    # Create a dummy image
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Test MobileNetV2
    extractor = CNNFeatureExtractor(model_name='MobileNetV2')
    features = extractor.extract_features(dummy_image)
    
    print(f"MobileNetV2 feature shape: {features.shape}")
    print(f"Feature vector sample: {features[:5]}")
    
    # Test image statistics
    stats = extract_image_statistics(dummy_image)
    print(f"\nImage statistics: {stats}")
