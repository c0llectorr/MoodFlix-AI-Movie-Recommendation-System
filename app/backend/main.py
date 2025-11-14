"""
Production-Ready FastAPI Backend for DeBERTa Emotion Detection
Optimized for 710MB model with efficient tokenization and inference
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Tokenizer
import uvicorn
import logging
import time
from typing import Dict, List, Optional
import gc
import psutil
import os
import json
import httpx
from dotenv import load_dotenv
import warnings

# Suppress HuggingFace deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== ENVIRONMENT SETUP ====================
load_dotenv()  # Load from .env file
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# ==================== CONFIGURATION ====================
class Config:
    MODEL_PATH = "./../../models/"  # Path to your saved model
    MAX_LENGTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8  # For batch predictions
    MAX_TEXT_LENGTH = 5000  # Character limit for input
    
    EMOTION_CLASSES = ['anger', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']
    
    # Emotion to Genre Mapping (4 genres per emotion)
    EMOTION_GENRE_MAP = {
        'anger': ['Action', 'Crime', 'Thriller', 'Revenge-Drama'],
        'fear': ['Horror', 'Thriller', 'Mystery', 'Supernatural'],
        'joy': ['Comedy', 'Adventure', 'Family', 'Animation', 'Musical'],
        'love': ['Romance', 'Rom-Com', 'Emotional Drama', 'Fantasy'],
        'neutral': ['Documentary', 'Drama', 'Biography', 'Slice-of-Life'],
        'sadness': ['Drama', 'Romance', 'Indie', 'Healing-Stories'],
        'surprise': ['Mystery', 'Sci-Fi', 'Fantasy', 'Twist-Thriller']
    }
    
    # Performance settings
    TORCH_THREADS = 4
    USE_HALF_PRECISION = False  # FP16 for GPU

config = Config()

# Set torch threads for CPU inference
torch.set_num_threads(config.TORCH_THREADS)

# ==================== MODEL DEFINITION ====================
class DeBERTaEmotionClassifier(nn.Module):
    """DeBERTa model for emotion classification"""
    def __init__(self, config_dict: Dict, num_labels: int):
        super().__init__()
        from transformers import DebertaV2Config
        deberta_config = DebertaV2Config(**config_dict)
        self.deberta = DebertaV2Model(deberta_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(deberta_config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# ==================== MODEL MANAGER ====================
class ModelManager:
    """Singleton class to manage model loading and inference"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.model = None
            self.tokenizer = None
            self.initialized = True
    
    def load_model(self):
        """Load model and tokenizer with error handling"""
        try:
            logger.info(f"Loading model on device: {config.DEVICE}")
            start_time = time.time()
            
            # Load config from local file
            logger.info("Loading model config...")
            config_path = os.path.join(config.MODEL_PATH, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found at {config_path}")
            
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Load tokenizer from local files or fallback to HuggingFace
            logger.info("Loading tokenizer...")
            tokenizer_files = {
                'vocab.json': os.path.join(config.MODEL_PATH, 'vocab.json'),
                'merges.txt': os.path.join(config.MODEL_PATH, 'merges.txt')
            }
            if os.path.exists(tokenizer_files['vocab.json']):
                logger.info("Loading tokenizer from local files...")
                self.tokenizer = DebertaV2Tokenizer(vocab_file=tokenizer_files['vocab.json'])
            else:
                logger.info("Downloading tokenizer from HuggingFace...")
                self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
            
            # Load model architecture from config
            logger.info("Loading model architecture from config...")
            self.model = DeBERTaEmotionClassifier(config_dict=model_config, num_labels=len(config.EMOTION_CLASSES))
            
            # Load trained weights: prefer full model file (safetensors) then fall back to checkpoint
            safetensors_path = os.path.join(config.MODEL_PATH, "model.safetensors")
            classifier_path = os.path.join(config.MODEL_PATH, "classifier.pt")

            # Helper to map keys that were saved without proper prefixes
            def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                """
                Normalize state dict keys to match the model architecture:
                - Add 'deberta.' prefix to encoder/embedding keys if missing
                - Add 'classifier.' prefix to weight/bias keys if they're classifier weights
                """
                if not state_dict:
                    return state_dict
                
                mapped = {}
                first_key = next(iter(state_dict.keys()), None)
                
                for k, v in state_dict.items():
                    # Check if this is a classifier weight/bias (2D/1D tensor from Linear layer)
                    # Classifier: weight [7, 768], bias [7]
                    if k in ['weight', 'bias']:
                        # Verify this is classifier-sized (num_classes=7, hidden_size=768)
                        if k == 'weight' and len(v.shape) == 2 and v.shape[0] == 7 and v.shape[1] == 768:
                            mapped['classifier.weight'] = v
                        elif k == 'bias' and len(v.shape) == 1 and v.shape[0] == 7:
                            mapped['classifier.bias'] = v
                        else:
                            mapped[k] = v
                    # Add deberta prefix to encoder/embedding keys if missing
                    elif k.startswith("deberta."):
                        # Already has prefix
                        mapped[k] = v
                    elif k.startswith("embeddings.") or k.startswith("encoder.") or k.startswith("rel_embeddings") or k.startswith("LayerNorm") or k.startswith("word_embeddings"):
                        # Encoder/embedding key without prefix
                        mapped[f"deberta.{k}"] = v
                    else:
                        # Keep as-is (dropout, etc.)
                        mapped[k] = v
                
                return mapped

            # 1) Try safetensors full-model file (fast and safe if present)
            if os.path.exists(safetensors_path):
                try:
                    logger.info("Loading full trained model from safetensors...")
                    try:
                        from safetensors.torch import load_file
                    except Exception as ie:
                        logger.error("safetensors package not available: %s", ie)
                        raise

                    state = load_file(safetensors_path)
                    # state is a dict of tensors; adjust keys if necessary
                    mapped = _normalize_state_dict_keys(state)
                    load_res = self.model.load_state_dict(mapped, strict=False)
                    logger.info("Loaded safetensors model (missing: %s, unexpected: %s)", getattr(load_res, 'missing_keys', []), getattr(load_res, 'unexpected_keys', []))
                except Exception as e:
                    logger.error("Error loading safetensors model: %s", e, exc_info=True)
                    # fall through to try checkpoint

            # 2) Fallback: try classic PyTorch checkpoint
            if os.path.exists(classifier_path):
                try:
                    logger.info("Loading trained checkpoint (torch) ...")
                    # In recent PyTorch versions the default weights_only may block some globals. Use weights_only=False
                    checkpoint = torch.load(classifier_path, map_location=config.DEVICE, weights_only=False)

                    # checkpoint might be a dict containing different keys depending on how it was saved
                    if isinstance(checkpoint, dict):
                        # Common key names used in training scripts
                        if 'model_state_dict' in checkpoint:
                            sd = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            sd = checkpoint['state_dict']
                        elif 'classifier_state_dict' in checkpoint or 'dropout_state_dict' in checkpoint:
                            # old style: only classifier saved
                            if 'classifier_state_dict' in checkpoint:
                                try:
                                    self.model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                                except Exception:
                                    # try flexible load with key normalization
                                    normalized = _normalize_state_dict_keys(checkpoint['classifier_state_dict'])
                                    self.model.load_state_dict(normalized, strict=False)
                            if 'dropout_state_dict' in checkpoint:
                                try:
                                    self.model.dropout.load_state_dict(checkpoint['dropout_state_dict'])
                                except Exception:
                                    logger.warning('Could not load dropout state dict')
                            sd = None
                        else:
                            sd = checkpoint

                        if sd:
                            # sd may contain keys without proper prefixes
                            mapped = _normalize_state_dict_keys(sd)
                            load_res = self.model.load_state_dict(mapped, strict=False)
                            logger.info("Loaded checkpoint (missing: %s, unexpected: %s)", getattr(load_res, 'missing_keys', []), getattr(load_res, 'unexpected_keys', []))
                    else:
                        # Not a dict: cannot handle
                        logger.warning("Checkpoint loaded but is not a dict, skipping")

                except Exception as e:
                    logger.error("Error loading checkpoint: %s", e, exc_info=True)
                    raise
            else:
                logger.warning(f"No trained model found at {safetensors_path} or {classifier_path}, using base model")
            
            # Move to device
            self.model.to(config.DEVICE)
            self.model.eval()
            
            # Apply half precision for GPU
            if config.USE_HALF_PRECISION:
                self.model.half()
                logger.info("Applied FP16 precision")
            
            # Optimize for inference
            if config.DEVICE == "cuda":
                torch.backends.cudnn.benchmark = True
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise HTTPException(status_code=500, detail=f"Model files not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """Run inference on input text"""
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model not loaded")
            
            start_time = time.time()
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(config.DEVICE)
            attention_mask = encoding['attention_mask'].to(config.DEVICE)
            
            # Apply half precision if enabled
            if config.USE_HALF_PRECISION:
                input_ids = input_ids.half().long()  # Convert back to long for embeddings
            
            # Inference
            with torch.cuda.amp.autocast(enabled=config.USE_HALF_PRECISION):
                logits = self.model(input_ids, attention_mask)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)
            
            # Convert to CPU and Python list (no numpy needed)
            probs_list = probs.cpu().float().tolist()[0]
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item()
            
            # Create emotion probability dict
            emotion_probs = {
                emotion: float(probs_list[i]) 
                for i, emotion in enumerate(config.EMOTION_CLASSES)
            }
            
            inference_time = time.time() - start_time
            
            return {
                "emotion": config.EMOTION_CLASSES[predicted_idx],
                "confidence": confidence_score,
                "all_probabilities": emotion_probs,
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# ==================== LIFESPAN MANAGEMENT ====================
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting application...")
    try:
        model_manager.load_model()
        logger.info("Application ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    model_manager.cleanup()
    logger.info("Application stopped")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Emotion Detection API",
    description="DeBERTa v3 based emotion detection for 7 emotion classes",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
FRONTEND_URL = 'https://moodflix-ai-nu.vercel.app'

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Local backend
        FRONTEND_URL,  # Production frontend
        "https://*.vercel.app",  # All Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=config.MAX_TEXT_LENGTH)
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time_ms: float
    text_length: int

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    memory_mb: float

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=10)
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError('At least one valid text required')
        if len(cleaned) > 10:
            raise ValueError('Maximum 10 texts allowed per batch')
        return cleaned

# ==================== MOVIE MODELS ====================
class MovieItem(BaseModel):
    id: int
    title: str
    poster_path: Optional[str] = None
    vote_average: float
    release_date: Optional[str] = None

class GenreMovies(BaseModel):
    genre: str
    movies: List[MovieItem]

class RecommendationResponse(BaseModel):
    emotion: str
    confidence: float
    recommendations: List[GenreMovies]

# ==================== ENDPOINTS ====================
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "emotions": "/emotions"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        memory_mb = psutil.Process().memory_info().rss / 1024 ** 2
        return HealthResponse(
            status="healthy",
            device=config.DEVICE,
            model_loaded=model_manager.model is not None,
            memory_mb=round(memory_mb, 2)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/emotions", response_model=Dict)
async def get_emotions():
    """Get list of supported emotions"""
    return {
        "emotions": config.EMOTION_CLASSES,
        "count": len(config.EMOTION_CLASSES)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """Predict emotion for single text"""
    try:
        logger.info(f"Prediction request: {len(request.text)} chars")
        
        result = model_manager.predict(request.text)
        
        return PredictionResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            inference_time_ms=result["inference_time_ms"],
            text_length=len(request.text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict_emotion(request: BatchPredictionRequest):
    """Predict emotions for multiple texts"""
    try:
        logger.info(f"Batch prediction request: {len(request.texts)} texts")
        
        results = []
        for text in request.texts:
            result = model_manager.predict(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "all_probabilities": result["all_probabilities"]
            })
        
        return {
            "count": len(results),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_movie_recommendations(request: PredictionRequest):
    """
    Predict emotion and fetch movie recommendations based on detected emotion.
    Maps emotion to genres and fetches 12 most popular movies per genre from TMDB.
    """
    try:
        logger.info(f"Recommendation request: {len(request.text)} chars")
        
        # 1. Detect emotion
        emotion_result = model_manager.predict(request.text)
        detected_emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]
        
        logger.info(f"Detected emotion: {detected_emotion} (confidence: {confidence})")
        
        # 2. Get mapped genres for this emotion
        genres = config.EMOTION_GENRE_MAP.get(detected_emotion, [])
        if not genres:
            raise HTTPException(status_code=400, detail=f"No genres mapped for emotion: {detected_emotion}")
        
        # 3. Fetch movies for these genres
        movies_by_genre = await fetch_movies_by_genres(genres, limit=12)
        
        # 4. Format response
        recommendations = []
        for genre in genres:
            if genre in movies_by_genre:
                recommendations.append(GenreMovies(
                    genre=genre,
                    movies=movies_by_genre[genre]
                ))
        
        return RecommendationResponse(
            emotion=detected_emotion,
            confidence=confidence,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

# ==================== ERROR HANDLERS ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

# ==================== TMDB MOVIE FETCHING ====================
async def fetch_movies_by_genres(genres: List[str], limit: int = 12) -> Dict[str, List[MovieItem]]:
    """
    Fetch movies from TMDB API for given genres.
    Returns dict with genre as key and list of MovieItem as value.
    """
    if not TMDB_API_KEY:
        logger.error("TMDB_API_KEY not configured")
        raise HTTPException(status_code=500, detail="Movie service not configured. Please set TMDB_API_KEY environment variable.")
    
    movies_by_genre = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for genre in genres:
            try:
                logger.info(f"Fetching movies for genre: {genre}")
                
                # Get genre ID from genre name
                genres_response = await client.get(
                    f"{TMDB_BASE_URL}/genre/movie/list",
                    params={"api_key": TMDB_API_KEY}
                )
                genres_response.raise_for_status()
                genres_data = genres_response.json()
                
                genre_id = None
                for g in genres_data.get("genres", []):
                    if g["name"].lower() == genre.lower():
                        genre_id = g["id"]
                        break
                
                if not genre_id:
                    logger.warning(f"Genre '{genre}' not found in TMDB")
                    continue
                
                # Fetch movies for this genre, sorted by popularity (descending)
                movies_response = await client.get(
                    f"{TMDB_BASE_URL}/discover/movie",
                    params={
                        "api_key": TMDB_API_KEY,
                        "with_genres": genre_id,
                        "sort_by": "popularity.desc",
                        "page": 1,
                        "language": "en-US"
                    }
                )
                movies_response.raise_for_status()
                movies_data = movies_response.json()
                
                # Parse movies
                movie_list = []
                for movie in movies_data.get("results", [])[:limit]:
                    movie_list.append(MovieItem(
                        id=movie.get("id"),
                        title=movie.get("title", "Unknown"),
                        poster_path=movie.get("poster_path"),
                        vote_average=movie.get("vote_average", 0.0),
                        release_date=movie.get("release_date", "N/A")
                    ))
                
                if movie_list:
                    movies_by_genre[genre] = movie_list
                    logger.info(f"Fetched {len(movie_list)} movies for genre: {genre}")
                else:
                    logger.warning(f"No movies found for genre: {genre}")
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching movies for {genre}: {e}")
            except Exception as e:
                logger.error(f"Error fetching movies for {genre}: {e}", exc_info=True)
    
    return movies_by_genre

# ==================== MAIN ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )