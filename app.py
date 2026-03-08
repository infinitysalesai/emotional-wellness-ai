from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import pipeline
import os
import tempfile
import uuid
import logging
from typing import Optional
import uvicorn

# ================= CONFIGURATION =================
app = FastAPI(
    title="Emotional Voice Audio Agent", 
    version="1.0.0",
    description="Pure audio analysis engine - extracts voice data for Lovable to process"
)

# Enable CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crisis keywords (just detection - no response generation)
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "want to die", "better off dead",
    "suicide", "ending it all", "no reason to live", "can't go on",
    "hurt myself", "self harm", "end it all", "want to disappear"
]

# ================= MODEL LOADING =================
logger.info("Loading audio analysis models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load speech-to-text model
try:
    logger.info("Loading speech-to-text model...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if device == "cuda" else -1
    )
    logger.info("✅ Speech-to-text model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load speech-to-text model: {e}")
    asr_pipeline = None

# ================= AUDIO ANALYSIS FUNCTIONS =================

def analyze_audio_features(audio_path: str) -> dict:
    """Extract ALL audio features - pure data extraction"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        features = {}
        
        # 1. PITCH ANALYSIS (Tonal intelligence)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            features['pitch_avg'] = float(np.mean(pitches))
            features['pitch_variance'] = float(np.std(pitches))
            features['pitch_max'] = float(np.max(pitches))
            features['pitch_min'] = float(np.min(pitches))
        else:
            features['pitch_avg'] = 0
            features['pitch_variance'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0
        
        # 2. ENERGY/VOLUME ANALYSIS
        rms = librosa.feature.rms(y=audio)[0]
        features['volume_avg'] = float(np.mean(rms))
        features['volume_max'] = float(np.max(rms))
        features['volume_variance'] = float(np.std(rms))
        
        # Determine volume level
        if features['volume_avg'] > 0.1:
            features['volume_level'] = "loud"
        elif features['volume_avg'] > 0.05:
            features['volume_level'] = "moderate"
        else:
            features['volume_level'] = "quiet"
        
        # Detect yelling (high volume + high variance)
        features['yelling_detected'] = features['volume_avg'] > 0.15 and features['volume_variance'] > 0.05
        features['whisper_detected'] = features['volume_avg'] < 0.02
        features['calm_speaking'] = 0.02 <= features['volume_avg'] <= 0.1 and features['volume_variance'] < 0.03
        
        # 3. SPEAKING RATE
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['speaking_rate'] = float(np.mean(zcr))
        
        # 4. PAUSE ANALYSIS
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = 0.01 * np.max(energy) if np.max(energy) > 0 else 0.01
        voice_frames = energy > threshold
        if len(voice_frames) > 0:
            features['pause_ratio'] = float(1.0 - (np.sum(voice_frames) / len(voice_frames)))
            features['pauses_frequent'] = features['pause_ratio'] > 0.3
        else:
            features['pause_ratio'] = 0
            features['pauses_frequent'] = False
        
        # 5. VOICE QUALITY (trembling detection)
        # Higher pitch variance + lower energy = trembling
        if features['pitch_variance'] > 50 and features['volume_avg'] < 0.08:
            features['voice_quality'] = "trembling"
        elif features['pitch_variance'] > 40:
            features['voice_quality'] = "emotional"
        elif features['volume_avg'] > 0.12:
            features['voice_quality'] = "strong"
        else:
            features['voice_quality'] = "steady"
        
        # 6. MELODIC CURVE (rising/falling tone)
        if len(pitches) > 5:
            first_pitches = np.mean(pitches[:5])
            last_pitches = np.mean(pitches[-5:])
            if last_pitches > first_pitches * 1.1:
                features['melodic_curve'] = "rising"
            elif last_pitches < first_pitches * 0.9:
                features['melodic_curve'] = "falling"
            else:
                features['melodic_curve'] = "flat"
        else:
            features['melodic_curve'] = "flat"
        
        # 7. SPEECH PACE
        if features['speaking_rate'] > 0.1:
            features['pace'] = "fast"
        elif features['speaking_rate'] > 0.06:
            features['pace'] = "normal"
        else:
            features['pace'] = "slow"
        
        # 8. HESITATION DETECTION
        features['hesitation_detected'] = features['pauses_frequent'] and features['pace'] == "slow"
        
        return features
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return {
            'pitch_avg': 0,
            'volume_level': 'unknown',
            'voice_quality': 'unknown',
            'error': str(e)
        }

def detect_emotional_cues(features: dict) -> dict:
    """Detect emotional cues from audio features - pure detection, no response"""
    
    emotion_data = {
        'primary': 'neutral',
        'secondary': None,
        'crying_detected': False,
        'laughter_detected': False,
        'anger_detected': False,
        'confidence': 0.5
    }
    
    # Crying detection (trembling voice + pauses + low-moderate volume)
    if (features.get('voice_quality') == 'trembling' and 
        features.get('pauses_frequent', False) and
        features.get('volume_level') in ['quiet', 'moderate']):
        emotion_data['crying_detected'] = True
        emotion_data['primary'] = 'sadness'
        emotion_data['confidence'] = 0.85
    
    # Anger detection (high volume, fast pace, high pitch variance)
    elif (features.get('yelling_detected', False) or
          (features.get('volume_level') == 'loud' and 
           features.get('pace') == 'fast' and
           features.get('pitch_variance', 0) > 45)):
        emotion_data['anger_detected'] = True
        emotion_data['primary'] = 'anger'
        emotion_data['confidence'] = 0.8
    
    # Laughter detection (rhythmic pitch variation, moderate volume)
    elif (features.get('pitch_variance', 0) > 30 and
          features.get('pace') == 'fast' and
          features.get('volume_level') == 'moderate'):
        # This is simplified - in production use a laughter detection model
        emotion_data['laughter_detected'] = False  # Placeholder
    
    # Anxiety detection (hesitation + trembling)
    elif (features.get('hesitation_detected', False) and
          features.get('voice_quality') == 'trembling'):
        emotion_data['primary'] = 'anxiety'
        emotion_data['confidence'] = 0.75
    
    # Sadness (slow pace, falling melodic curve, quiet)
    elif (features.get('pace') == 'slow' and
          features.get('melodic_curve') == 'falling' and
          features.get('volume_level') in ['quiet', 'moderate']):
        emotion_data['primary'] = 'sadness'
        emotion_data['confidence'] = 0.7
    
    # Excitement (fast pace, rising melodic curve, higher pitch)
    elif (features.get('pace') == 'fast' and
          features.get('melodic_curve') == 'rising' and
          features.get('pitch_avg', 0) > 150):
        emotion_data['primary'] = 'excitement'
        emotion_data['confidence'] = 0.65
    
    return emotion_data

def detect_crisis_in_text(text: str) -> bool:
    """Pure crisis keyword detection - no response generation"""
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def transcribe_audio(audio_path: str) -> str:
    """Convert speech to text"""
    try:
        if asr_pipeline:
            result = asr_pipeline(audio_path)
            return result.get('text', '')
        else:
            return "[Voice message]"
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "[Voice message]"

# ================= API ENDPOINTS =================

@app.get("/")
async def root():
    """API info - pure audio agent"""
    return {
        "service": "Emotional Voice Audio Agent",
        "version": "1.0.0",
        "status": "online",
        "role": "I only extract audio data. Lovable does all the thinking.",
        "endpoints": {
            "/health": "GET - Health check",
            "/analyze": "POST - Extract ALL audio data (STT + emotion + tone + volume)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": asr_pipeline is not None,
        "device": device,
        "message": "Audio agent ready to extract data"
    }

@app.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    PURE AUDIO ANALYSIS - Returns structured data only
    
    Extracts:
    - What was said (STT)
    - Emotional state (crying, laughing, anger)
    - Tonal intelligence (pitch, quality)
    - Volume analysis (yelling, whispering)
    - Speech patterns (pauses, pace)
    - Crisis flag (keywords only)
    
    Lovable does ALL thinking/response generation
    """
    try:
        logger.info(f"Received audio: {audio.filename}")
        
        # Save audio temporarily
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # 1. Speech-to-Text (What was said)
        transcript = transcribe_audio(tmp_path)
        
        # 2. Extract ALL audio features
        audio_features = analyze_audio_features(tmp_path)
        
        # 3. Detect emotional cues
        emotion_data = detect_emotional_cues(audio_features)
        
        # 4. Crisis detection (keywords only)
        crisis_detected = detect_crisis_in_text(transcript)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Return PURE DATA - Lovable does the thinking
        response_data = {
            "success": True,
            
            # 1. WHAT WAS SAID
            "text": transcript,
            
            # 2. EMOTIONAL STATE
            "emotion": {
                "primary": emotion_data['primary'],
                "secondary": emotion_data['secondary'],
                "crying_detected": emotion_data['crying_detected'],
                "laughter_detected": emotion_data['laughter_detected'],
                "anger_detected": emotion_data['anger_detected'],
                "confidence": emotion_data['confidence']
            },
            
            # 3. TONAL INTELLIGENCE
            "tone": {
                "pitch_avg": audio_features.get('pitch_avg', 0),
                "pitch_variance": audio_features.get('pitch_variance', 0),
                "voice_quality": audio_features.get('voice_quality', 'unknown'),
                "melodic_curve": audio_features.get('melodic_curve', 'flat')
            },
            
            # 4. VOLUME ANALYSIS
            "volume": {
                "level": audio_features.get('volume_level', 'unknown'),
                "db_avg": audio_features.get('volume_avg', 0),
                "yelling_detected": audio_features.get('yelling_detected', False),
                "whisper_detected": audio_features.get('whisper_detected', False),
                "calm_speaking": audio_features.get('calm_speaking', False)
            },
            
            # 5. SPEECH PATTERNS
            "speech_patterns": {
                "pace": audio_features.get('pace', 'unknown'),
                "pauses": "frequent" if audio_features.get('pauses_frequent', False) else "normal",
                "hesitation_detected": audio_features.get('hesitation_detected', False)
            },
            
            # 6. CRISIS FLAG (Lovable decides what to do)
            "crisis_detected": crisis_detected,
            
            # Session
            "session_id": session_id
        }
        
        logger.info(f"Analysis complete - Emotion: {emotion_data['primary']}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

# ================= LAUNCH =================
if __name__ != "__main__":
    # When running on Hugging Face
    pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)