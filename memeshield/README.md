# MemeShield 🛡️

**AI-Powered Hate Speech Detection in Memes**

MemeShield is a multimodal AI system that detects hate speech in memes by analyzing both visual content and extracted text. It combines OCR (Optical Character Recognition), CNN-based image analysis, and NLP techniques to provide accurate hate speech detection.

![MemeShield](https://img.shields.io/badge/MemeShield-AI%20Powered-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=flat-square)

---

## 🌟 Features

- **Multimodal Analysis**: Combines text and image analysis for better accuracy
- **OCR Processing**: Extracts text from meme images using Tesseract
- **CNN Feature Extraction**: Uses MobileNetV2/ResNet50 for visual features
- **Real-time Predictions**: Fast inference with confidence scores
- **Modern UI**: Clean, responsive web interface
- **History Tracking**: MySQL database for storing analysis results
- **REST API**: JSON endpoints for integration

---

## 📁 Project Structure

```
memeshield/
│
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
│
├── model/                  # Trained model files (add your models here)
│
├── static/
│   ├── css/
│   │   └── style.css      # Main stylesheet
│   └── js/
│       └── app.js         # Frontend JavaScript
│
├── templates/
│   ├── index.html         # Home page
│   └── history.html       # History page
│
├── uploads/                # Uploaded meme images
│
├── utils/
│   ├── __init__.py        # Utils package
│   ├── ocr.py             # OCR text extraction
│   ├── preprocess.py      # Text & image preprocessing
│   ├── cnn_model.py       # CNN feature extractor
│   └── fusion_model.py    # Multimodal fusion classifier
│
└── database/
    └── schema.sql         # MySQL database schema
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- MySQL Server 8.0+
- Tesseract OCR

### Step 1: Clone & Navigate

```bash
cd memeshield
```

### Step 2: Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add to PATH (optional)

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Configure Environment

```bash
# Copy example env file
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env with your settings
```

Edit `.env` file:
```env
SECRET_KEY=your-super-secret-key-change-this
DEBUG=True

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=memeshield
MYSQL_PORT=3306

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Step 5: Set Up MySQL Database

```bash
# Connect to MySQL
mysql -u root -p

# Run the schema
source database/schema.sql
```

Or using Python:
```bash
python -c "from app import init_database; init_database()"
```

### Step 6: Run the Application

```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/upload` | POST | Upload meme image |
| `/predict` | POST | Run AI analysis |
| `/history` | GET | View prediction history |
| `/api/history` | GET | Get history as JSON |
| `/api/stats` | GET | Get statistics |
| `/delete/<id>` | DELETE | Delete a record |

### Example API Usage

**Upload & Analyze:**
```javascript
// Upload file
const formData = new FormData();
formData.append('file', imageFile);

const uploadResponse = await fetch('/upload', {
    method: 'POST',
    body: formData
});
const uploadData = await uploadResponse.json();

// Get prediction
const predictResponse = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: uploadData.filename })
});
const result = await predictResponse.json();
console.log(result.prediction, result.confidence);
```

---

## 🧠 AI Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   OCR       │────▶│   Text      │
│   Image     │     │  (Tesseract)│     │ Preprocess  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │   Image     │            │
                    │ Preprocess  │            │
                    └──────┬──────┘            │
                           │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐
                    │   CNN       │     │   Text      │
                    │ (MobileNet) │     │  Encoder    │
                    └──────┬──────┘     └──────┬──────┘
                           │                   │
                    ┌──────▼───────────────────▼──────┐
                    │        Multimodal Fusion         │
                    │      (Late Fusion Strategy)      │
                    └──────────────┬──────────────────┘
                                   │
                            ┌──────▼──────┐
                            │ Classification│
                            │ Hate/Non-Hate │
                            └─────────────┘
```

---

## 🎯 Model Training (Advanced)

To train the model on your own dataset:

```python
from utils.cnn_model import CNNFeatureExtractor
from utils.fusion_model import MemeClassifier

# Prepare your data
# meme_data = [(text, image_features), ...]
# labels = [0, 1, 0, 1, ...]  # 0 = non-hate, 1 = hate

# Train
classifier = MemeClassifier()
history = classifier.train(meme_data, labels)

# Save
classifier.save('model/trained_model')
```

---

## 🔮 Future Enhancements

- [ ] **CLIP Integration**: Use OpenAI's CLIP for better vision-language understanding
- [ ] **BERT Embeddings**: Replace simple text encoder with BERT
- [ ] **Multilingual Support**: Add Hindi/Marathi text detection
- [ ] **Real-time API**: WebSocket for live moderation
- [ ] **Browser Extension**: Chrome/Firefox extension for social media
- [ ] **Mobile App**: React Native mobile application

---

## 🐛 Troubleshooting

### Common Issues

**1. Tesseract not found:**
```
pytesseract.pytesseract.TesseractNotFoundError
```
→ Set correct path in `.env`: `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

**2. MySQL connection error:**
```
mysql.connector.errors.ProgrammingError
```
→ Check MySQL is running and credentials in `.env` are correct

**3. TensorFlow GPU issues:**
```
Could not load dynamic library 'cudart64_*.dll'
```
→ Install CUDA toolkit or use CPU version: `pip install tensorflow-cpu`

**4. NLTK data not found:**
```
LookupError: Resource punkt not found
```
→ Run: `python -c "import nltk; nltk.download('all')"`

---

## 📝 License

This project is for educational purposes. Please use responsibly.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using TensorFlow, Flask, and modern web technologies.**
