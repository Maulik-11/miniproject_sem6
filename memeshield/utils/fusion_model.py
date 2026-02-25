"""
MemeShield Multimodal Fusion Model
Combines text and image features for hate speech classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TextEncoder:
    """
    Text encoder that converts text to embeddings.
    Uses a simple architecture that can be upgraded to BERT-ready structure.
    """
    
    def __init__(self, vocab_size=10000, max_length=128, embedding_dim=128):
        """
        Initialize the text encoder.
        
        Args:
            vocab_size (int): Size of vocabulary.
            max_length (int): Maximum sequence length.
            embedding_dim (int): Dimension of text embeddings.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.model = None
        self.is_fitted = False
        
        self._build_model()
    
    def _build_model(self):
        """Build the text encoding model."""
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        x = layers.Embedding(
            self.vocab_size, 
            self.embedding_dim,
            input_length=self.max_length
        )(inputs)
        
        # Bidirectional LSTM for sequence understanding
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.embedding_dim, activation='relu')(x)
        
        self.model = Model(inputs, outputs)
    
    def fit_tokenizer(self, texts):
        """
        Fit the tokenizer on a corpus of texts.
        
        Args:
            texts (list): List of text strings.
        """
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
    
    def encode(self, text):
        """
        Encode a single text to a feature vector.
        
        Args:
            text (str): Input text.
        
        Returns:
            numpy.ndarray: Text embedding vector.
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_dim)
        
        # Tokenize
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Encode
        embedding = self.model.predict(padded, verbose=0)
        return embedding.flatten()
    
    def encode_batch(self, texts):
        """
        Encode a batch of texts.
        
        Args:
            texts (list): List of text strings.
        
        Returns:
            numpy.ndarray: Batch of text embeddings.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        embeddings = self.model.predict(padded, verbose=0)
        return embeddings
    
    def save(self, path):
        """Save the encoder model and tokenizer."""
        self.model.save(os.path.join(path, 'text_encoder.h5'))
        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load(self, path):
        """Load the encoder model and tokenizer."""
        self.model = keras.models.load_model(os.path.join(path, 'text_encoder.h5'))
        with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.is_fitted = True


class MultimodalFusionModel:
    """
    Multimodal fusion model that combines text and image features.
    Uses late fusion strategy for combining modalities.
    """
    
    def __init__(self, text_dim=128, image_dim=1280, hidden_dim=256, num_classes=2):
        """
        Initialize the fusion model.
        
        Args:
            text_dim (int): Dimension of text features.
            image_dim (int): Dimension of image features (from CNN).
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
        """
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the multimodal fusion model."""
        # Text input branch
        text_input = layers.Input(shape=(self.text_dim,), name='text_input')
        text_features = layers.Dense(128, activation='relu')(text_input)
        text_features = layers.Dropout(0.3)(text_features)
        text_features = layers.Dense(64, activation='relu')(text_features)
        
        # Image input branch
        image_input = layers.Input(shape=(self.image_dim,), name='image_input')
        image_features = layers.Dense(256, activation='relu')(image_input)
        image_features = layers.Dropout(0.3)(image_features)
        image_features = layers.Dense(64, activation='relu')(image_features)
        
        # Fusion: Concatenate features
        fused = layers.Concatenate()([text_features, image_features])
        
        # Fusion layers
        x = layers.Dense(self.hidden_dim, activation='relu')(fused)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Build model
        self.model = Model(
            inputs=[text_input, image_input],
            outputs=outputs
        )
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Multimodal fusion model built.")
    
    def train(self, text_features, image_features, labels, 
              validation_split=0.2, epochs=20, batch_size=32):
        """
        Train the fusion model.
        
        Args:
            text_features: Text feature matrix.
            image_features: Image feature matrix.
            labels: One-hot encoded labels.
            validation_split: Fraction of data for validation.
            epochs: Number of training epochs.
            batch_size: Batch size.
        
        Returns:
            Training history.
        """
        history = self.model.fit(
            {'text_input': text_features, 'image_input': image_features},
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        return history
    
    def predict(self, text_features, image_features):
        """
        Make a prediction using both modalities.
        
        Args:
            text_features: Text feature vector.
            image_features: Image feature vector.
        
        Returns:
            tuple: (predicted_class, confidence)
        """
        # Ensure correct shape
        if len(text_features.shape) == 1:
            text_features = np.expand_dims(text_features, axis=0)
        if len(image_features.shape) == 1:
            image_features = np.expand_dims(image_features, axis=0)
        
        predictions = self.model.predict(
            {'text_input': text_features, 'image_input': image_features},
            verbose=0
        )
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, float(confidence)
    
    def save(self, path):
        """Save the model."""
        self.model.save(path)
    
    def load(self, path):
        """Load a saved model."""
        self.model = keras.models.load_model(path)


class MemeClassifier:
    """
    Main classifier that orchestrates the full pipeline.
    Combines text encoding, image features, and fusion model.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the meme classifier.
        
        Args:
            model_path (str): Path to load pre-trained models.
        """
        self.text_encoder = TextEncoder()
        self.fusion_model = None
        self.is_trained = False
        
        # Initialize with a simple rule-based classifier
        # In production, load trained models
        self._init_default_classifier()
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _init_default_classifier(self):
        """Initialize a default/demo classifier."""
        # Create fusion model but don't train it
        self.fusion_model = MultimodalFusionModel(
            text_dim=128,
            image_dim=1280,  # MobileNetV2 output dim
            hidden_dim=256
        )
        
        # Fit tokenizer with some sample hate speech vocabulary
        # In production, this should be trained on actual data
        sample_texts = [
            "hate speech example text",
            "normal meme text",
            "offensive content",
            "funny joke",
            "discriminatory language",
            "happy meme"
        ]
        self.text_encoder.fit_tokenizer(sample_texts)
    
    def predict(self, text, image_features):
        """
        Predict if a meme contains hate speech.
        
        Args:
            text (str): Extracted and preprocessed text from meme.
            image_features (numpy.ndarray): CNN features from image.
        
        Returns:
            tuple: (prediction_label, confidence)
        """
        try:
            # Encode text
            text_features = self.text_encoder.encode(text)
            
            # Use hybrid approach: rule-based + model
            confidence = self._hybrid_predict(text, text_features, image_features)
            
            # Determine label
            if confidence > 0.5:
                label = 'hate'
            else:
                label = 'non-hate'
                confidence = 1 - confidence  # Confidence for non-hate
            
            return label, confidence
        
        except Exception as e:
            print(f"Prediction error: {e}")
            # Default to non-hate with low confidence
            return 'non-hate', 0.5
    
    def _hybrid_predict(self, text, text_features, image_features):
        """
        Hybrid prediction using both rule-based and learned features.
        
        This is a demonstration implementation. In production:
        1. Train the fusion model on labeled hate speech data
        2. Use proper text embeddings (BERT, etc.)
        3. Fine-tune CNN for hate speech visual patterns
        """
        # Rule-based component: check for hate speech indicators
        hate_indicators = [
            'hate', 'kill', 'die', 'death', 'stupid', 'idiot', 'ugly',
            'racist', 'sexist', 'disgusting', 'worthless', 'loser',
            'terrorist', 'criminal', 'scum', 'trash', 'garbage',
            'moron', 'retard', 'freak', 'psycho'
        ]
        
        text_lower = text.lower() if text else ""
        
        # Count hate indicators in text
        hate_score = 0
        for indicator in hate_indicators:
            if indicator in text_lower:
                hate_score += 1
        
        # Normalize rule-based score
        rule_score = min(hate_score / 3, 1.0)  # Cap at 1.0
        
        # If model is trained, combine with model prediction
        if self.is_trained:
            try:
                # Ensure correct dimensions
                if len(text_features.shape) == 1:
                    text_features = np.expand_dims(text_features, axis=0)
                if len(image_features.shape) == 1:
                    image_features = np.expand_dims(image_features, axis=0)
                
                model_pred = self.fusion_model.model.predict(
                    {'text_input': text_features, 'image_input': image_features},
                    verbose=0
                )
                model_score = model_pred[0][1]  # Probability of hate class
                
                # Combine rule-based and model scores
                combined_score = 0.3 * rule_score + 0.7 * model_score
                return combined_score
            except:
                pass
        
        # Return rule-based score with some noise for demo
        noise = np.random.uniform(-0.1, 0.1)
        return np.clip(rule_score + noise + 0.1, 0, 1)
    
    def train(self, meme_data, labels):
        """
        Train the classifier on labeled meme data.
        
        Args:
            meme_data: List of (text, image_features) tuples.
            labels: List of labels (0 for non-hate, 1 for hate).
        """
        # Extract texts for tokenizer training
        texts = [item[0] for item in meme_data]
        self.text_encoder.fit_tokenizer(texts)
        
        # Encode all texts
        text_features = np.array([
            self.text_encoder.encode(text) for text, _ in meme_data
        ])
        
        # Stack image features
        image_features = np.array([img_feat for _, img_feat in meme_data])
        
        # One-hot encode labels
        labels_onehot = keras.utils.to_categorical(labels, num_classes=2)
        
        # Train fusion model
        history = self.fusion_model.train(
            text_features, image_features, labels_onehot
        )
        
        self.is_trained = True
        return history
    
    def save(self, path):
        """Save all model components."""
        os.makedirs(path, exist_ok=True)
        self.text_encoder.save(path)
        self.fusion_model.save(os.path.join(path, 'fusion_model.h5'))
    
    def load(self, path):
        """Load saved model components."""
        try:
            self.text_encoder.load(path)
            self.fusion_model.load(os.path.join(path, 'fusion_model.h5'))
            self.is_trained = True
        except Exception as e:
            print(f"Error loading models: {e}")


class AttentionFusion(layers.Layer):
    """
    Attention-based fusion layer for future improvements.
    Can be used to weight text vs image features dynamically.
    """
    
    def __init__(self, units=64, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.units = units
        self.W_text = layers.Dense(units)
        self.W_image = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, text_features, image_features):
        # Compute attention scores
        text_score = self.V(tf.nn.tanh(self.W_text(text_features)))
        image_score = self.V(tf.nn.tanh(self.W_image(image_features)))
        
        # Softmax over modalities
        scores = tf.concat([text_score, image_score], axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Weight features
        weighted_text = text_features * attention_weights[:, 0:1]
        weighted_image = image_features * attention_weights[:, 1:2]
        
        # Fused output
        fused = tf.concat([weighted_text, weighted_image], axis=-1)
        return fused, attention_weights


# Future-ready: CLIP-style model skeleton
class CLIPStyleModel:
    """
    Skeleton for CLIP-style vision-language model.
    For future implementation with actual CLIP weights.
    """
    
    def __init__(self):
        self.text_encoder = None
        self.image_encoder = None
        self.temperature = 0.07
    
    def encode_text(self, text):
        """Encode text to embedding (placeholder)."""
        pass
    
    def encode_image(self, image):
        """Encode image to embedding (placeholder)."""
        pass
    
    def similarity(self, text_embed, image_embed):
        """Compute cosine similarity between embeddings."""
        text_norm = text_embed / np.linalg.norm(text_embed)
        image_norm = image_embed / np.linalg.norm(image_embed)
        return np.dot(text_norm, image_norm)


# Test function
if __name__ == "__main__":
    print("Testing Multimodal Fusion Model...")
    
    # Create dummy data
    dummy_text_features = np.random.rand(1, 128).astype(np.float32)
    dummy_image_features = np.random.rand(1, 1280).astype(np.float32)
    
    # Test fusion model
    fusion = MultimodalFusionModel()
    pred_class, confidence = fusion.predict(dummy_text_features, dummy_image_features)
    print(f"Fusion model prediction: class={pred_class}, confidence={confidence:.4f}")
    
    # Test full classifier
    classifier = MemeClassifier()
    label, conf = classifier.predict("test hate text", dummy_image_features.flatten())
    print(f"Classifier prediction: {label}, confidence={conf:.4f}")
