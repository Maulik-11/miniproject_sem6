"""
MemeShield Preprocessing Module
Handles text and image preprocessing for the AI pipeline.
"""

import re
import string
import numpy as np
from PIL import Image
import cv2

# NLP libraries
import nltk

# Initialize lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Download required NLTK data (run once)
def ensure_nltk_data():
    """Download NLTK data if not present."""
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

# Try to get stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                     'very', 'just', 'i', 'me', 'my', 'myself', 'we', 'our',
                     'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                     'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                     'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                     'this', 'that', 'these', 'those', 'am'])

# Common hate speech related terms for feature detection
HATE_INDICATORS = {
    'slurs', 'offensive', 'racist', 'sexist', 'homophobic',
    'discriminatory', 'violent', 'threatening', 'derogatory'
}

# Image settings
IMAGE_SIZE = (224, 224)


def clean_text(text):
    """
    Clean raw text by removing special characters and normalizing.
    
    Args:
        text (str): Raw text extracted from image.
    
    Returns:
        str: Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def remove_stopwords(text):
    """
    Remove common stopwords from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with stopwords removed.
    """
    if not text:
        return ""
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return ' '.join(filtered_words)


def tokenize_text(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text.
    
    Returns:
        list: List of tokens.
    """
    if not text:
        return []
    
    try:
        # Try NLTK word_tokenize
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        # Fallback to regex-based tokenization
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens


def lemmatize_text(text):
    """
    Lemmatize words in text to their base form.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Lemmatized text.
    """
    if not text:
        return ""
    
    tokens = tokenize_text(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)


def preprocess_text(text, remove_stops=True, lemmatize=True):
    """
    Full text preprocessing pipeline.
    
    Args:
        text (str): Raw text extracted from image.
        remove_stops (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to lemmatize words.
    
    Returns:
        str: Fully preprocessed text.
    """
    if not text:
        return ""
    
    # Step 1: Clean the text
    cleaned = clean_text(text)
    
    # Step 2: Remove stopwords (optional)
    if remove_stops:
        cleaned = remove_stopwords(cleaned)
    
    # Step 3: Lemmatize (optional)
    if lemmatize:
        cleaned = lemmatize_text(cleaned)
    
    return cleaned


def text_to_features(text, max_length=512, vocab_size=10000):
    """
    Convert text to numerical features for the model.
    Simple bag-of-words style encoding.
    
    Args:
        text (str): Preprocessed text.
        max_length (int): Maximum sequence length.
        vocab_size (int): Size of vocabulary.
    
    Returns:
        numpy.ndarray: Numerical feature vector.
    """
    if not text:
        return np.zeros(max_length)
    
    # Simple character-level encoding
    # In production, use proper embeddings (BERT, Word2Vec, etc.)
    tokens = tokenize_text(text)
    
    # Create simple hash-based features
    features = np.zeros(max_length)
    for i, token in enumerate(tokens[:max_length]):
        # Simple hash to convert word to number
        features[i] = hash(token) % vocab_size
    
    return features


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocess image for CNN feature extraction.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size (width, height) for resizing.
    
    Returns:
        numpy.ndarray: Preprocessed image array ready for CNN.
    """
    try:
        # Load image using PIL
        image = Image.open(image_path)
        
        # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        # Return a blank image as fallback
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)


def preprocess_image_cv2(image_path, target_size=IMAGE_SIZE):
    """
    Alternative image preprocessing using OpenCV.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size (width, height) for resizing.
    
    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    try:
        # Read image with OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    except Exception as e:
        print(f"OpenCV preprocessing error: {e}")
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)


def augment_image(image_array, augmentation_type='random'):
    """
    Apply data augmentation to image.
    Useful for training the model.
    
    Args:
        image_array (numpy.ndarray): Image array of shape (H, W, 3).
        augmentation_type (str): Type of augmentation.
    
    Returns:
        numpy.ndarray: Augmented image array.
    """
    if augmentation_type == 'flip_horizontal':
        return np.fliplr(image_array)
    
    elif augmentation_type == 'flip_vertical':
        return np.flipud(image_array)
    
    elif augmentation_type == 'rotate':
        # Rotate 90 degrees
        return np.rot90(image_array)
    
    elif augmentation_type == 'brightness':
        # Adjust brightness
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image_array * factor, 0, 1)
    
    elif augmentation_type == 'random':
        # Apply random augmentation
        augmentations = ['flip_horizontal', 'brightness', 'none']
        chosen = np.random.choice(augmentations)
        if chosen == 'none':
            return image_array
        return augment_image(image_array, chosen)
    
    return image_array


def extract_text_features(text):
    """
    Extract statistical features from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        dict: Dictionary of text features.
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'uppercase_ratio': 0,
            'exclamation_count': 0,
            'question_count': 0
        }
    
    words = text.split()
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?')
    }


# Test function
if __name__ == "__main__":
    # Test text preprocessing
    sample_text = """
    THIS IS A TEST MEME TEXT!!!
    With some URLs: https://example.com
    And emails: test@email.com
    Numbers: 12345
    Special chars: @#$%^&*
    """
    
    print("Original text:")
    print(sample_text)
    
    print("\nCleaned text:")
    print(clean_text(sample_text))
    
    print("\nFully preprocessed text:")
    print(preprocess_text(sample_text))
    
    print("\nText features:")
    print(extract_text_features(sample_text))
