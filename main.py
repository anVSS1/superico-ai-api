# Superico AI Streamer - Production API
# Deploy this to Render.com, Railway, or any cloud service

import os
import logging
import hashlib
from datetime import datetime
from typing import Optional
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# AI imports
from groq import Groq
from TTS.api import TTS
import torch
import librosa
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Superico AI Streamer API", 
    version="2.0",
    description="Trilingual Moroccan AI Streamer with Voice Cloning"
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
groq_client = None
tts_model = None
reference_audio_path = None

class ChatRequest(BaseModel):
    message: str
    username: Optional[str] = "anonymous"
    stream_context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    tts_language: str
    audio_url: Optional[str] = None
    timestamp: datetime
    personality_mode: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global groq_client, tts_model, reference_audio_path
    
    logger.info("🚀 Starting Superico AI Streamer Production API...")
    
    # Initialize Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        logger.info("✅ Groq LLaMA-3 client ready")
    else:
        logger.error("❌ GROQ_API_KEY not found!")
    
    # Initialize XTTS with your voice model
    try:
        logger.info("🎤 Loading XTTS with Superico's voice...")
        
        # Setup reference audio path
        reference_audio_path = "./voice_model/reference_audio.wav"
        
        if os.path.exists(reference_audio_path):
            # Load XTTS model
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            logger.info("✅ XTTS loaded with Superico's voice!")
        else:
            logger.warning("⚠️ Voice model not found, using text-only mode")
            
    except Exception as e:
        logger.error(f"❌ XTTS loading failed: {e}")
        logger.info("📝 Running in text-only mode")

@app.get("/")
async def health_check():
    """API health check"""
    return {
        "status": "🔥 Superico AI is LIVE!",
        "service": "Trilingual Moroccan AI Streamer",
        "version": "2.0",
        "features": ["Darija", "English", "French", "Voice Cloning"],
        "models": {
            "llm": "LLaMA-3 via Groq" if groq_client else "Offline",
            "tts": "XTTS v2 + Superico Voice" if tts_model else "Text-only",
            "voice_model": "Loaded" if reference_audio_path and os.path.exists(reference_audio_path) else "Missing"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_superico(request: ChatRequest):
    """Main chat endpoint - this is what OBS calls"""
    try:
        if not groq_client:
            raise HTTPException(status_code=503, detail="AI model not available")
        
        logger.info(f"💬 {request.username}: {request.message}")
        
        # Detect language
        detected_lang = detect_language(request.message)
        
        # Generate AI response
        ai_response = await generate_ai_response(
            request.message, 
            request.username, 
            detected_lang,
            request.stream_context
        )
        
        # Generate voice if possible
        audio_url = None
        if tts_model and reference_audio_path:
            audio_url = await generate_voice_async(ai_response, detected_lang)
        
        response = ChatResponse(
            response=ai_response,
            detected_language=detected_lang,
            tts_language=map_language_to_tts(detected_lang),
            audio_url=audio_url,
            timestamp=datetime.now(),
            personality_mode="superico_streamer"
        )
        
        logger.info(f"🤖 Response: {ai_response[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    audio_path = f"./audio_cache/{filename}"
    if os.path.exists(audio_path):
        return FileResponse(
            audio_path, 
            media_type="audio/wav",
            headers={"Cache-Control": "max-age=3600"}
        )
    raise HTTPException(status_code=404, detail="Audio not found")

def detect_language(text: str) -> str:
    """Enhanced language detection"""
    text_lower = text.lower()
    
    # Arabic script
    if any('\u0600' <= char <= '\u06FF' for char in text):
        return "darija"
    
    # Darija patterns
    darija_patterns = [
        "wash", "kayn", "kifash", "l7al", "wallah", "aywa", "bzzaf", 
        "3afak", "7bibi", "9ahwa", "3andi", "ghadi", "daba", "aji"
    ]
    if any(pattern in text_lower for pattern in darija_patterns):
        return "darija"
    
    # French patterns
    french_patterns = [
        "bonjour", "salut", "comment", "ça va", "merci", "très", 
        "oui", "non", "français", "magnifique", "incroyable"
    ]
    if any(pattern in text_lower for pattern in french_patterns):
        return "french"
    
    # Mixed language check
    has_darija = any(p in text_lower for p in darija_patterns[:5])
    has_english = any(p in text_lower for p in ["how", "what", "game", "play", "good"])
    has_french = any(p in text_lower for p in french_patterns[:3])
    
    if sum([has_darija, has_english, has_french]) > 1:
        return "mixed"
    
    return "english"

async def generate_ai_response(message: str, username: str, language: str, context: str = None) -> str:
    """Generate AI response using Groq LLaMA-3"""
    
    system_prompts = {
        "darija": '''You are Superico, a popular Moroccan Twitch streamer. Respond in Darija with Franco-Arabic.

PERSONALITY: Energetic, funny, gaming-focused Moroccan streamer
LANGUAGE: Respond in Darija using Latin script with numbers (3=ع, 7=ح, 9=ق, etc.)
STYLE: 1-2 sentences, gaming slang, Moroccan expressions

EXAMPLES:
- "wash kayn a 5ouya! kifash l7al m3ak?"
- "aywa had l3ba zwin bzzaf, ready for action!"
- "wallah nta 9ad pro f had game!"''',

        "french": '''You are Superico, a popular Moroccan Twitch streamer. Respond in French with Moroccan flair.

PERSONALITY: Energetic, funny, gaming-focused Moroccan streamer  
LANGUAGE: Respond in French with occasional Darija expressions
STYLE: 1-2 sentences, gaming terms, energetic

EXAMPLES:
- "Salut mon pote! Ça va bien ou quoi?"
- "Ce jeu est vraiment génial wallah!"
- "Tu es très fort à ce game, respect!"''',

        "english": '''You are Superico, a popular Moroccan Twitch streamer. Respond in English with Moroccan personality.

PERSONALITY: Energetic, funny, gaming-focused Moroccan streamer
LANGUAGE: Respond in English with some Darija expressions  
STYLE: 1-2 sentences, gaming slang, authentic

EXAMPLES:
- "Yo what's good bro! Ready to game?"
- "That was absolutely sick wallah!"
- "You're really skilled aywa, respect!"''',

        "mixed": '''You are Superico, a popular Moroccan Twitch streamer. Mix languages naturally.

PERSONALITY: Energetic, funny, gaming-focused Moroccan streamer
LANGUAGE: Mix Darija, English, and French naturally
STYLE: 1-2 sentences, switch languages fluidly

EXAMPLES:
- "Salut bro! wash kayn? Ready for gaming?"
- "C'est vraiment sick wallah! Tu es bon!"
- "Aywa let's go! This game est magnifique!"'''
    }
    
    system_prompt = system_prompts.get(language, system_prompts["english"])
    
    user_prompt = f"Viewer '{username}' says: {message}"
    if context:
        user_prompt += f"\nStream context: {context}"
    
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=150,
        temperature=0.8
    )
    
    return response.choices[0].message.content.strip()

async def generate_voice_async(text: str, language: str) -> Optional[str]:
    """Generate voice using XTTS with Superico's voice"""
    if not tts_model or not reference_audio_path:
        return None
    
    try:
        # Create audio cache directory
        os.makedirs("./audio_cache", exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"superico_{timestamp}.wav"
        output_path = f"./audio_cache/{filename}"
        
        # Map language for TTS
        tts_lang = map_language_to_tts(language)
        
        # Generate voice with Superico's reference audio
        tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=tts_lang,
            file_path=output_path
        )
        
        return f"/audio/{filename}"
        
    except Exception as e:
        logger.error(f"❌ Voice generation error: {e}")
        return None

def map_language_to_tts(detected_lang: str) -> str:
    """Map detected language to TTS language codes"""
    mapping = {
        "darija": "ar",
        "french": "fr", 
        "english": "en",
        "mixed": "en"
    }
    return mapping.get(detected_lang, "en")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
