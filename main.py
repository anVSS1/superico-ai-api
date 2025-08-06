# Superico AI Streamer - Production API (Simplified)
# Deploy this to Render.com, Railway, or any cloud service

import os
import logging
import re
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import tempfile
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# AI imports
from groq import Groq

# ElevenLabs Voice Cloning imports
from elevenlabs import generate, clone, set_api_key
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
groq_client = None
voice_model_loaded = False
elevenlabs_voice_id = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global groq_client, voice_model_loaded, elevenlabs_voice_id
    
    logger.info("🚀 Starting Superico AI Streamer Production API...")
    
    # Initialize Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    logger.info(f"API Key found: {'Yes' if groq_api_key else 'No'}")
    if groq_api_key:
        logger.info(f"API Key starts with: {groq_api_key[:10]}...")
        try:
            # Clear any proxy-related env vars that might interfere
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    logger.info(f"Removing proxy env var: {var}")
                    del os.environ[var]
            
            # Clean Groq client initialization
            groq_client = Groq(
                api_key=groq_api_key
            )
            logger.info("✅ Groq client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Groq initialization failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            groq_client = None
    else:
        logger.error("❌ GROQ_API_KEY not found!")
        groq_client = None
    
    # Initialize ElevenLabs Voice Cloning
    try:
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_api_key:
            logger.info("🎤 Initializing ElevenLabs voice cloning...")
            set_api_key(elevenlabs_api_key)
            
            # Check if reference audio exists for voice cloning
            reference_path = "reference_audio.wav"
            if os.path.exists(reference_path):
                logger.info(f"✅ Found reference audio: {reference_path}")
                
                # Clone voice from reference audio
                logger.info("🎯 Cloning Superico's voice...")
                voice = clone(
                    name="Superico_Streamer",
                    description="Superico's cloned voice for streaming",
                    files=[reference_path]
                )
                elevenlabs_voice_id = voice.voice_id
                voice_model_loaded = True
                logger.info(f"✅ Voice cloned successfully! Voice ID: {elevenlabs_voice_id}")
                
            else:
                logger.warning(f"⚠️ Reference audio not found at {reference_path}")
                logger.info("📝 Using default ElevenLabs voice")
                elevenlabs_voice_id = "pNInz6obpgDQGcFmaJgB"  # Default voice
                voice_model_loaded = True
                
        else:
            logger.warning("⚠️ ELEVENLABS_API_KEY not found - voice generation disabled")
            voice_model_loaded = False
            
    except Exception as e:
        logger.error(f"❌ ElevenLabs initialization failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        voice_model_loaded = False
    
    yield
    
    # Cleanup (if needed)
    logger.info("🛑 Shutting down Superico AI...")

# FastAPI app with lifespan
app = FastAPI(
    title="Superico AI Streamer API", 
    version="2.0",
    description="Trilingual Moroccan AI Streamer with Voice Cloning",
    lifespan=lifespan
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    username: Optional[str] = "anonymous"
    stream_context: Optional[str] = "general"

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    audio_url: Optional[str] = None
    timestamp: datetime
    personality_mode: str

@app.get("/")
async def health_check():
    """API health check"""
    return {
        "status": "🔥 Superico AI is LIVE!",
        "service": "Trilingual Moroccan AI Streamer",
        "version": "2.0",
        "features": ["Darija", "English", "French", "AI Chat"],
        "models": {
            "llm": "LLaMA-3 via Groq" if groq_client else "Offline",
            "tts": "ElevenLabs Voice Cloning" if voice_model_loaded else "Offline",
            "voice_model": "Superico's Cloned Voice" if elevenlabs_voice_id else "Not loaded"
        }
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment"""
    # Test Groq client creation directly
    groq_test_result = None
    groq_error = None
    try:
        from groq import Groq
        groq_key = os.getenv("GROQ_API_KEY")
        # Use exact same initialization as lifespan
        test_client = Groq(
            api_key=groq_key
        )
        groq_test_result = "SUCCESS - Client created"
        
        # Try a simple API call
        try:
            models = test_client.models.list()
            groq_test_result += f" - API accessible, {len(models.data)} models available"
        except Exception as api_e:
            groq_test_result += f" - Client created but API call failed: {str(api_e)}"
            
    except Exception as groq_e:
        groq_error = str(groq_e)
        groq_test_result = f"FAILED - {groq_error}"
    
    return {
        "env_vars": {
            "GROQ_API_KEY_exists": bool(os.getenv("GROQ_API_KEY")),
            "GROQ_API_KEY_length": len(os.getenv("GROQ_API_KEY", "")),
            "GROQ_API_KEY_prefix": (os.getenv("GROQ_API_KEY", "")[:10] + "...") if os.getenv("GROQ_API_KEY") else "None",
            "PORT": os.getenv("PORT", "Not set"),
            "all_env_keys": [k for k in os.environ.keys() if 'GROQ' in k.upper()]
        },
        "groq_client_status": str(type(groq_client)) if groq_client else "None",
        "groq_test_creation": groq_test_result,
        "groq_error": groq_error,
        "groq_version": getattr(__import__('groq'), '__version__', 'unknown')
    }

@app.get("/health")
async def detailed_health():
    """Detailed health check"""
    global voice_model_loaded
    return {
        "status": "healthy",
        "groq_connected": bool(groq_client),
        "voice_model_loaded": voice_model_loaded,
        "supported_languages": ["darija", "english", "french", "franco-arabic"],
        "timestamp": datetime.now().isoformat()
    }

async def generate_voice(text: str, language: str) -> Optional[str]:
    """Generate voice audio using ElevenLabs voice cloning"""
    try:
        if not voice_model_loaded or not elevenlabs_voice_id:
            logger.warning("⚠️ Voice model not loaded, skipping audio generation")
            return None
            
        logger.info(f"🎤 Generating Superico's cloned voice for: {text[:50]}...")
        
        # Preprocess text to fix numbers and pronunciation
        processed_text = preprocess_text_for_speech(text, language)
        
        # Generate speech using ElevenLabs with cloned voice
        audio_data = generate(
            text=processed_text,
            voice=elevenlabs_voice_id,
            model="eleven_multilingual_v2"  # Supports multiple languages
        )
        
        # Save audio data to file
        audio_id = str(uuid.uuid4())
        wav_path = f"/tmp/superico_voice_{audio_id}.wav"
        
        with open(wav_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"✅ Superico's voice generated: {wav_path}")
        return f"/audio/{audio_id}"
        
    except Exception as e:
        logger.error(f"❌ Voice generation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def preprocess_text_for_speech(text: str, language: str) -> str:
    """Preprocess text to improve speech synthesis"""
    import re
    
    # Remove or replace problematic characters
    processed = text
    
    if language == "darija" or language == "franco-arabic":
        # Convert Arabic numerals to spoken form in Darija
        processed = re.sub(r'\b(\d+)\b', lambda m: convert_number_to_darija(int(m.group(1))), processed)
        
        # Fix common Darija pronunciation issues
        processed = processed.replace("3", "ع")  # Replace 3 with proper Arabic letter
        processed = processed.replace("7", "ح")  # Replace 7 with proper Arabic letter
        processed = processed.replace("9", "ق")  # Replace 9 with proper Arabic letter
        
    elif language == "english":
        # Keep numbers as digits for English (reads them properly)
        pass
        
    elif language == "french":
        # Keep numbers as digits for French
        pass
    
    return processed

def convert_number_to_darija(num: int) -> str:
    """Convert numbers to Darija words"""
    darija_numbers = {
        0: "صفر", 1: "واحد", 2: "جوج", 3: "تلاتة", 4: "ربعة", 5: "خمسة",
        6: "ستة", 7: "سبعة", 8: "تمنية", 9: "تسعة", 10: "عشرة",
        11: "حداش", 12: "طناش", 13: "تلطاش", 14: "ربعطاش", 15: "خمسطاش",
        16: "سطاش", 17: "سبعطاش", 18: "تمنطاش", 19: "تسعطاش", 20: "عشرين"
    }
    
    if num in darija_numbers:
        return darija_numbers[num]
    elif num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        if tens == 20:
            return f"عشرين و {darija_numbers[ones]}" if ones > 0 else "عشرين"
        else:
            return str(num)  # Fallback to digit for complex numbers
    else:
        return str(num)  # Fallback to digit for large numbers

def detect_language(text: str) -> str:
    """Detect language of input text"""
    text_lower = text.lower()
    
    # Darija indicators
    darija_words = ['wash', 'kayn', 'walakin', 'bzaf', 'hna', 'hada', 'dyal', 'fin', 'kifash', 'mnin']
    
    # French indicators  
    french_words = ['salut', 'bonjour', 'comment', 'ça', 'va', 'tu', 'vous', 'merci', 'oui', 'non']
    
    # Count indicators
    darija_count = sum(1 for word in darija_words if word in text_lower)
    french_count = sum(1 for word in french_words if word in text_lower)
    
    # Franco-Arabic (mix of languages)
    has_arabic_script = bool(re.search(r'[\u0600-\u06FF]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    
    if has_arabic_script and has_latin:
        return "franco-arabic"
    elif darija_count > 0:
        return "darija"
    elif french_count > 0:
        return "french"
    else:
        return "english"

async def generate_ai_response(message: str, username: str, language: str, context: str) -> str:
    """Generate AI response using Groq"""
    
    # Superico's personality system prompt
    system_prompt = f"""You are Superico, a popular Moroccan Twitch streamer and gamer. You're chatting with your viewers during a live stream.

Your personality:
- Funny, energetic, and entertaining
- Mix languages naturally (Darija, French, English) 
- Use gaming slang and Moroccan expressions
- Always engaging and interactive with chat
- Make jokes and references to gaming/streaming culture

Current context: {context}
Detected language: {language}

Respond in the same language as the viewer, but feel free to mix languages naturally like a real Moroccan streamer would.
Keep responses short (1-2 sentences) for live streaming.
Be entertaining and authentic to Superico's personality."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{username} says: {message}"}
            ],
            max_tokens=150,
            temperature=0.8
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        
        # Fallback responses by language
        fallbacks = {
            "darija": "Salam khouya! Ana Superico, wash kayn chi haja?",
            "french": "Salut! Je suis Superico, comment ça va?", 
            "english": "Hey! I'm Superico, what's up?",
            "franco-arabic": "Yo bro! Ana Superico, wash ready ndir game?"
        }
        
        return fallbacks.get(language, "Hey! I'm Superico, what's up?")

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
        
        # Generate voice audio
        audio_url = await generate_voice(ai_response, detected_lang)
        
        response = ChatResponse(
            response=ai_response,
            detected_language=detected_lang,
            audio_url=audio_url,
            timestamp=datetime.now(),
            personality_mode="superico_streamer"
        )
        
        logger.info(f"🤖 Response: {ai_response}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """Serve generated audio files"""
    try:
        audio_path = f"/tmp/superico_voice_{audio_id}.wav"
        if os.path.exists(audio_path):
            return FileResponse(
                audio_path,
                media_type="audio/wav",
                filename=f"superico_{audio_id}.wav"
            )
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        logger.error(f"❌ Audio serving error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
