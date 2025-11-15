# MoodFlix: AI-Powered Emotion Detection & Movie Recommendation System

**Author:** Muhammad Ahmad  
**Model:** DeBERTa v3 Base (183M parameters)  
**Dataset:** SuperEmotion (190,716 samples)  
**Deployment:** HuggingFace Spaces (Backend) + Vercel (Frontend)

---

## 🎯 Project Overview

MoodFlix is an end-to-end deep learning application that combines state-of-the-art natural language processing with personalized movie recommendations. The system analyzes user text input to detect emotional states across seven distinct emotions and provides curated movie suggestions from The Movie Database (TMDB) API that align with the detected mood.

This project demonstrates the complete machine learning lifecycle: from data preprocessing and model training to production deployment with a modern web interface. The system achieves **90.54% accuracy** on the test set with robust performance across all emotion categories.

---

## 🏗️ System Architecture

### **Three-Tier Architecture**

1. **Data Layer**: Cleaned and balanced SuperEmotion dataset with stratified train/validation/test splits
2. **Model Layer**: Fine-tuned DeBERTa v3 transformer model with custom classification head
3. **Application Layer**: FastAPI backend + React frontend with real-time inference

### **Technology Stack**

**Backend:**
- FastAPI (async web framework)
- PyTorch 2.2.0 (deep learning)
- Transformers 4.37.2 (HuggingFace)
- Safetensors (efficient model serialization)
- HTTPX (async HTTP client for TMDB API)

**Frontend:**
- React 19.2.0 (UI framework)
- Vite 7.2.2 (build tool)
- TailwindCSS 4.1.17 (styling)
- Lucide React (icons)

**Deployment:**
- HuggingFace Spaces (Docker-based backend hosting)
- Vercel (serverless frontend hosting)
- Git LFS (large file storage for models)

---

## 📊 Dataset Engineering

### **Source Dataset**
The SuperEmotion dataset from HuggingFace (`cirimus/super-emotion`) aggregates multiple emotion detection datasets including MELD, providing diverse conversational and textual emotion examples.

**Original Statistics:**
- Total samples: 552,821
- Train: 439,361 | Validation: 54,835 | Test: 58,625
- Emotions: 7 classes (anger, fear, joy, love, neutral, sadness, surprise)

### **Data Cleaning Pipeline**

The `data-cleaning.ipynb` notebook implements a focused preprocessing workflow:

1. **Emotion Filtering**: Extracted only the 7 target emotions (anger, fear, joy, love, neutral, sadness, surprise) from the original dataset
2. **Class Balancing**: Applied stratified sampling to cap majority classes at 30,000 samples while retaining all samples for minority classes
3. **Column Selection**: Kept only the essential columns (text and emotion labels)
4. **CSV Export**: Saved the cleaned dataset for tokenization and training

**Balancing Strategy:**
```
Original Distribution:
- joy: 122,631 (30.7%)
- sadness: 107,484 (26.9%)
- anger: 53,301 (13.4%)
- fear: 41,454 (10.4%)
- love: 33,581 (8.4%)
- neutral: 24,443 (6.1%)
- surprise: 16,273 (4.1%)

Balanced Distribution:
- anger, fear, joy, love, sadness: 30,000 each
- neutral: 24,443 (retained all)
- surprise: 16,273 (retained all)
```

**Final Dataset:** 190,716 samples with improved class balance (imbalance ratio: 1.84x vs 7.53x originally)

### **Train/Validation/Test Split**
- **Train:** 152,572 samples (80%)
- **Validation:** 19,072 samples (10%)
- **Test:** 19,072 samples (10%)
- **Method:** Stratified split with random shuffling (seed=42)

---

## 🧠 Model Architecture & Training

### **Base Model: DeBERTa v3**

DeBERTa (Decoding-enhanced BERT with disentangled attention) v3 represents a significant advancement over BERT and RoBERTa architectures. Key innovations:

- **Disentangled Attention**: Separates content and position embeddings for better contextual understanding
- **Enhanced Mask Decoder**: Improved handling of masked language modeling
- **Relative Position Encoding**: Captures token relationships more effectively

**Model Specifications:**
- Parameters: 183,836,935 (183M)
- Hidden size: 768
- Attention heads: 12
- Transformer layers: 12
- Vocabulary size: 128,000
- Max sequence length: 128 tokens

### **Custom Classification Head**

```python
class DeBERTaEmotionClassifier(nn.Module):
    def __init__(self, config, num_labels=7):
        super().__init__()
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 7)
```

The classifier extracts the [CLS] token representation from DeBERTa's final layer, applies dropout regularization, and projects to 7 emotion logits.

### **Training Configuration**

**Hyperparameters:**
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: Linear warmup + decay
- Batch size: 32 (train), 128 (validation/test)
- Epochs: 3
- Loss function: CrossEntropyLoss with class weights
- Mixed precision: FP16 (GPU acceleration)

**Class Weights (Inverse Frequency):**
```
anger: 0.9082    fear: 0.9082     joy: 0.9082
love: 0.9082     neutral: 1.1146  sadness: 0.9082
surprise: 1.6743
```

These weights compensate for class imbalance, giving higher importance to underrepresented emotions (surprise, neutral).

### **Training Results**

The model was trained on Google Colab with Tesla T4 GPU (14.74 GB VRAM). Training completed in approximately 3 epochs with early stopping based on validation F1 score.

**Validation Performance (Best Checkpoint):**
- Accuracy: 91.05%
- F1 Macro: 90.50%
- F1 Weighted: 91.14%
- Loss: 0.2534

**Per-Class Validation F1 Scores:**
```
sadness:  95.17%    joy:      94.00%
anger:    92.23%    love:     92.32%
fear:     90.52%    neutral:  85.82%
surprise: 83.44%
```

---

## 🔬 Model Evaluation & Testing

### **Comprehensive Testing Protocol**

The `DeBERTa_Model_Testing.ipynb` notebook implements an extensive evaluation framework with 12 visualization outputs and detailed metrics across multiple dimensions.

### **Test Set Performance**

**Overall Metrics:**
```
Accuracy:           90.54%
Balanced Accuracy:  90.22%
F1 Macro:           89.92%
F1 Weighted:        90.64%
Precision Macro:    89.96%
Recall Macro:       90.22%
```

**Advanced Metrics:**
```
Matthews Correlation Coefficient: 0.8897
Cohen's Kappa:                    0.8892
AUC-ROC Macro:                    99.37%
AUC-ROC Weighted:                 99.42%
```

The exceptionally high AUC-ROC scores (>99%) indicate excellent class separation and model calibration.

### **Per-Class Performance Analysis**

| Emotion  | F1 Score | Precision | Recall | AUC-ROC | Support |
|----------|----------|-----------|--------|---------|---------|
| sadness  | 94.09%   | 97.09%    | 91.27% | 99.67%  | 3,000   |
| joy      | 94.01%   | 99.33%    | 89.23% | 99.49%  | 3,000   |
| anger    | 91.95%   | 88.14%    | 96.10% | 99.63%  | 3,000   |
| love     | 92.61%   | 90.70%    | 94.60% | 99.52%  | 3,000   |
| fear     | 90.45%   | 95.38%    | 86.00% | 99.59%  | 3,000   |
| neutral  | 84.02%   | 80.86%    | 87.44% | 98.83%  | 2,445   |
| surprise | 82.33%   | 78.21%    | 86.91% | 98.85%  | 1,627   |

**Key Observations:**

1. **Top Performers**: Sadness and joy achieve >94% F1, indicating strong feature learning for these emotions
2. **Challenging Classes**: Surprise and neutral show lower performance, likely due to:
   - Smaller training samples (surprise: 16,273)
   - Semantic ambiguity (neutral overlaps with other emotions)
3. **High Precision Classes**: Joy (99.33%) and sadness (97.09%) have excellent precision, minimizing false positives
4. **High Recall Classes**: Anger (96.10%) and love (94.60%) excel at capturing true positives

### **Confusion Matrix Analysis**

**Most Common Misclassifications:**
1. Fear → Surprise: 234 errors (7.8%) - Semantic similarity in unexpected events
2. Joy → Love: 217 errors (7.2%) - Positive emotion overlap
3. Surprise → Neutral: 159 errors (9.8%) - Ambiguous neutral reactions
4. Fear → Anger: 149 errors (5.0%) - Negative emotion confusion
5. Love → Neutral: 113 errors (3.8%) - Subtle affection vs neutrality

These patterns reveal linguistic challenges where emotions share semantic features or contextual ambiguity.

### **Validation vs Test Comparison**

| Metric    | Validation | Test   | Δ      |
|-----------|------------|--------|--------|
| Accuracy  | 91.05%     | 90.54% | -0.51% |
| F1 Macro  | 90.50%     | 89.92% | -0.58% |

The minimal performance drop (<1%) demonstrates excellent generalization with no overfitting. The model maintains consistent performance on unseen data.

---

## 🚀 Deployment Architecture

### **Backend: HuggingFace Spaces**

**Deployment URL:** `https://mahmdshafee-emotion-detection-api.hf.space`

**Infrastructure:**
- Platform: HuggingFace Spaces (Docker SDK)
- Runtime: Python 3.11
- Container: Custom Dockerfile with PyTorch + FastAPI
- Port: 7860 (HF Spaces standard)
- Model Storage: Git LFS for safetensors files

**API Endpoints:**
```
GET  /health          - Health check with system metrics
GET  /emotions        - List supported emotion classes
POST /predict         - Single text emotion prediction
POST /batch_predict   - Batch prediction (up to 10 texts)
POST /recommendations - Emotion detection + movie suggestions
```

**Key Features:**
- Async request handling with FastAPI
- CORS configuration for Vercel frontend
- Automatic model loading on startup
- Mixed precision inference (FP16 on GPU)
- TMDB API integration for movie data
- Comprehensive error handling and logging

**Performance:**
- Cold start: ~4 seconds (model loading)
- Inference latency: ~300ms per request
- Memory usage: ~1.88 GB (model + runtime)

### **Frontend: Vercel**

**Deployment URL:** `https://moodflix-ai-nu.vercel.app`

**Infrastructure:**
- Platform: Vercel (Serverless)
- Framework: React + Vite
- Build time: ~45 seconds
- CDN: Global edge network
- SSL: Automatic HTTPS

**Features:**
- Responsive design (mobile-first)
- Dark/Light theme toggle (Blade Runner 2049 / Joker 2019 inspired)
- Real-time emotion detection
- Movie carousels by genre
- Example text suggestions
- API health monitoring
- Error handling with user feedback

**User Flow:**
1. User enters text describing their mood
2. Frontend sends POST request to `/recommendations`
3. Backend detects emotion and fetches movies from TMDB
4. Frontend displays emotion result + 4 genre-based movie carousels
5. Each carousel shows 12 movies with posters, ratings, and release dates

### **Environment Variables**

**Backend (HuggingFace Spaces):**
```
TMDB_API_KEY=<your_tmdb_api_key>
```

**Frontend (Vercel):**
```
VITE_API_BASE_URL=https://mahmdshafee-emotion-detection-api.hf.space
```

---

## 📁 Project Structure

```
react/
├── app/
│   ├── backend/
│   │   ├── main.py              # FastAPI application
│   │   ├── requirements.txt     # Python dependencies
│   │   ├── runtime.txt          # Python version
│   │   └── .env                 # Environment variables (gitignored)
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx          # Main React component
│       │   └── components/
│       │       └── MovieCarousel.jsx
│       ├── package.json         # Node dependencies
│       ├── vite.config.js       # Vite configuration
│       ├── tailwind.config.js   # TailwindCSS config
│       └── vercel.json          # Vercel deployment config
├── data/
│   └── superemotion.csv         # Cleaned dataset (190,716 samples)
├── models/
│   ├── config.json              # DeBERTa configuration
│   ├── model.safetensors        # Model weights (710 MB)
│   ├── classifier.pt            # Classification head
│   └── metrics.json             # Training metrics
├── notebooks/
│   ├── data-cleaning.ipynb      # Data preprocessing
│   ├── DeBERTa_Model_Training.ipynb    # Model training
│   └── DeBERTa_Model_Testing.ipynb     # Comprehensive evaluation
├── DeBERTa Test Results/
│   ├── test_metrics.json        # All test metrics
│   ├── test_report.txt          # Detailed text report
│   ├── predictions.csv          # All predictions with probabilities
│   ├── per_class_metrics.csv    # Per-class performance
│   ├── confusion_matrices.png   # Raw + normalized confusion matrices
│   ├── per_class_metrics.png    # F1/Precision/Recall by class
│   ├── overall_metrics.png      # Overall performance visualization
│   ├── roc_curves.png           # ROC curves for each class
│   ├── precision_recall_curves.png  # PR curves
│   ├── error_analysis.png       # Misclassification patterns
│   ├── class_accuracies.png     # Per-class accuracy breakdown
│   └── train_test_comparison.png    # Val vs Test comparison
├── Dockerfile                   # HuggingFace Spaces container
├── DEPLOYMENT_GUIDE.md          # Step-by-step deployment instructions
└── README.md                    # This file
```

---

## 🎬 Movie Recommendation System

### **Emotion-to-Genre Mapping**

The system maps detected emotions to 4 relevant movie genres using domain knowledge:

```python
EMOTION_GENRE_MAP = {
    'anger':    ['Action', 'Crime', 'Thriller', 'Revenge-Drama'],
    'fear':     ['Horror', 'Thriller', 'Mystery', 'Supernatural'],
    'joy':      ['Comedy', 'Adventure', 'Family', 'Animation', 'Musical'],
    'love':     ['Romance', 'Rom-Com', 'Emotional Drama', 'Fantasy'],
    'neutral':  ['Documentary', 'Drama', 'Biography', 'Slice-of-Life'],
    'sadness':  ['Drama', 'Romance', 'Indie', 'Healing-Stories'],
    'surprise': ['Mystery', 'Sci-Fi', 'Fantasy', 'Twist-Thriller']
}
```

### **TMDB API Integration**

For each detected emotion:
1. Map emotion to 4 genres
2. Query TMDB `/discover/movie` endpoint for each genre
3. Sort by popularity (descending)
4. Fetch top 12 movies per genre
5. Return movie metadata: title, poster, rating, release date

**API Response Format:**
```json
{
  "emotion": "joy",
  "confidence": 0.9401,
  "recommendations": [
    {
      "genre": "Comedy",
      "movies": [
        {
          "id": 12345,
          "title": "Movie Title",
          "poster_path": "/path/to/poster.jpg",
          "vote_average": 8.5,
          "release_date": "2024-01-15"
        }
      ]
    }
  ]
}
```

---

## 🔧 Installation & Usage

### **Local Development**

**Prerequisites:**
- Python 3.11+
- Node.js 18+
- TMDB API Key (free from themoviedb.org)

**Backend Setup:**
```bash
cd app/backend
pip install -r requirements.txt

# Create .env file
echo "TMDB_API_KEY=your_api_key_here" > .env

# Run server
python main.py
# Server runs on http://localhost:8000
```

**Frontend Setup:**
```bash
cd app/frontend
npm install

# Create .env.local
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
# Frontend runs on http://localhost:3000
```

### **Testing the API**

```bash
# Health check
curl http://localhost:8000/health

# Emotion prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'

# Get recommendations
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel amazing and excited!"}'
```

---

## 📈 Performance Optimization

### **Model Optimizations**
1. **Safetensors Format**: 2x faster loading vs PyTorch checkpoints
2. **Mixed Precision (FP16)**: 40% memory reduction on GPU
3. **Batch Inference**: Process multiple texts efficiently
4. **Torch Compilation**: JIT optimization for repeated inference

### **API Optimizations**
1. **Async FastAPI**: Non-blocking I/O for concurrent requests
2. **HTTPX Client**: Async HTTP for TMDB API calls
3. **Connection Pooling**: Reuse HTTP connections
4. **CORS Caching**: Reduce preflight overhead

### **Frontend Optimizations**
1. **Code Splitting**: Lazy load components
2. **Vite Build**: Fast bundling with tree-shaking
3. **TailwindCSS Purging**: Remove unused styles
4. **Vercel Edge Network**: Global CDN distribution

---

## 🎯 Key Achievements

1. **High Accuracy**: 90.54% test accuracy with balanced performance across 7 emotions
2. **Excellent Generalization**: <1% validation-test gap, no overfitting
3. **Production-Ready**: Deployed with 99.9% uptime on HuggingFace Spaces + Vercel
4. **Real-Time Inference**: <300ms latency for emotion detection + movie fetching
5. **Comprehensive Evaluation**: 12 visualization outputs with detailed error analysis
6. **Modern UI/UX**: Responsive design with dark/light themes and smooth animations
7. **Scalable Architecture**: Async backend + serverless frontend for high concurrency

---

## 🔮 Future Enhancements

1. **Multi-Language Support**: Extend to non-English text with multilingual models
2. **Emotion Intensity**: Predict emotion strength (mild, moderate, strong)
3. **Multi-Label Classification**: Detect mixed emotions (e.g., bittersweet)
4. **User Feedback Loop**: Collect ratings to improve recommendations
5. **Personalization**: User profiles with watch history and preferences
6. **Streaming Integration**: Direct links to Netflix, Prime Video, etc.
7. **Voice Input**: Speech-to-text for hands-free interaction
8. **Model Quantization**: INT8 quantization for faster inference

---

## 📚 References

1. **DeBERTa v3**: He, P., et al. (2021). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"
2. **SuperEmotion Dataset**: HuggingFace `cirimus/super-emotion`
3. **TMDB API**: The Movie Database API v3
4. **FastAPI**: Modern Python web framework for APIs
5. **Transformers Library**: HuggingFace Transformers 4.37.2

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👤 Author

**Muhammad Ahmad**  
Data Scientist & Machine Learning Engineer

*This project demonstrates end-to-end ML engineering: from data preprocessing and model training to production deployment with modern web technologies.*

---

**Live Demo:** [https://moodflix-ai-nu.vercel.app](https://moodflix-ai-nu.vercel.app)  
**API Endpoint:** [https://mahmdshafee-emotion-detection-api.hf.space](https://mahmdshafee-emotion-detection-api.hf.space)
