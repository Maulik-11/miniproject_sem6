"""
MemeShield OCR Module
Extracts text from meme images using Tesseract OCR.
"""

import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Import config for Tesseract path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
    # Set Tesseract command path (Windows)
    if os.path.exists(Config.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
except ImportError:
    pass


def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text detection
    # This helps with memes that have various backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply morphological operations to clean up text
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return processed


def extract_text_from_image(image_path, lang='eng'):
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file.
        lang (str): Language code for OCR. Default is 'eng'.
                   Use 'eng+hin' for English and Hindi support.
    
    Returns:
        str: Extracted text from the image.
    """
    try:
        # Try multiple preprocessing methods and combine results
        extracted_texts = []
        
        # Method 1: Direct extraction with PIL
        try:
            pil_image = Image.open(image_path)
            text1 = pytesseract.image_to_string(pil_image, lang=lang)
            if text1.strip():
                extracted_texts.append(text1.strip())
        except Exception as e:
            print(f"PIL extraction failed: {e}")
        
        # Method 2: Preprocessed image extraction
        try:
            preprocessed = preprocess_image_for_ocr(image_path)
            text2 = pytesseract.image_to_string(preprocessed, lang=lang)
            if text2.strip():
                extracted_texts.append(text2.strip())
        except Exception as e:
            print(f"Preprocessed extraction failed: {e}")
        
        # Method 3: Inverted image (for white text on dark background)
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            text3 = pytesseract.image_to_string(inverted, lang=lang)
            if text3.strip():
                extracted_texts.append(text3.strip())
        except Exception as e:
            print(f"Inverted extraction failed: {e}")
        
        # Combine and deduplicate extracted texts
        if extracted_texts:
            # Use the longest extraction result as it's likely most complete
            combined_text = max(extracted_texts, key=len)
            return combined_text
        
        return ""
    
    except Exception as e:
        print(f"OCR extraction error: {e}")
        return ""


def extract_text_with_confidence(image_path, lang='eng'):
    """
    Extract text from image with confidence scores.
    
    Args:
        image_path (str): Path to the image file.
        lang (str): Language code for OCR.
    
    Returns:
        tuple: (extracted_text, average_confidence)
    """
    try:
        pil_image = Image.open(image_path)
        
        # Get detailed OCR data with confidence scores
        ocr_data = pytesseract.image_to_data(
            pil_image, lang=lang, output_type=pytesseract.Output.DICT
        )
        
        words = []
        confidences = []
        
        for i, word in enumerate(ocr_data['text']):
            if word.strip():
                conf = int(ocr_data['conf'][i])
                if conf > 0:  # Only include words with positive confidence
                    words.append(word)
                    confidences.append(conf)
        
        extracted_text = ' '.join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return extracted_text, avg_confidence / 100  # Normalize to 0-1
    
    except Exception as e:
        print(f"Confidence extraction error: {e}")
        return "", 0.0


def detect_text_regions(image_path):
    """
    Detect regions in the image that contain text.
    Useful for memes with text in specific areas.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        list: List of bounding boxes [(x, y, w, h), ...] containing text.
    """
    try:
        pil_image = Image.open(image_path)
        
        # Get bounding box data
        boxes = pytesseract.image_to_boxes(pil_image)
        
        regions = []
        for box in boxes.splitlines():
            b = box.split()
            if len(b) >= 5:
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                regions.append((x, y, w - x, h - y))
        
        return regions
    
    except Exception as e:
        print(f"Text region detection error: {e}")
        return []


def extract_text_multilingual(image_path, languages=['eng', 'hin', 'mar']):
    """
    Extract text supporting multiple languages.
    Useful for detecting Hindi/Marathi text in memes.
    
    Args:
        image_path (str): Path to the image file.
        languages (list): List of language codes to use.
    
    Returns:
        str: Combined extracted text from all languages.
    """
    lang_string = '+'.join(languages)
    return extract_text_from_image(image_path, lang=lang_string)


# Test function
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing OCR on: {test_image}")
        
        text = extract_text_from_image(test_image)
        print(f"\nExtracted Text:\n{text}")
        
        text_conf, confidence = extract_text_with_confidence(test_image)
        print(f"\nText with confidence: {text_conf}")
        print(f"Average confidence: {confidence:.2%}")
    else:
        print("Usage: python ocr.py <image_path>")
