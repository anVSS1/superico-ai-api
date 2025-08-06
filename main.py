# Superico AI Streamer - Production API (Simplified)
# Deploy this to Render.com, Railway, or any cloud service

import os
import logging
import re
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# AI imports
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
groq_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global groq_client
    
    logger.info("🚀 Starting Superico AI Streamer Production API...")
    
    # Initialize Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    logger.info(f"API Key found: {'Yes' if groq_api_key else 'No'}")
    if groq_api_key:
        logger.info(f"API Key starts with: {groq_api_key[:10]}...")
        try:
            groq_client = Groq(api_key=groq_api_key)
            logger.info("✅ Groq client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Groq initialization failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.error("❌ GROQ_API_KEY not found!")
    
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
            "tts": "Coming soon - deploy first!",
            "voice_model": "Ready for TTS integration"
        }
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment"""
    return {
        "env_vars": {
            "GROQ_API_KEY_exists": bool(os.getenv("GROQ_API_KEY")),
            "GROQ_API_KEY_length": len(os.getenv("GROQ_API_KEY", "")),
            "GROQ_API_KEY_prefix": (os.getenv("GROQ_API_KEY", "")[:10] + "...") if os.getenv("GROQ_API_KEY") else "None",
            "PORT": os.getenv("PORT", "Not set"),
            "all_env_keys": [k for k in os.environ.keys() if 'GROQ' in k.upper()]
        },
        "groq_client_status": str(type(groq_client)) if groq_client else "None"
    }

@app.get("/health")
async def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "groq_connected": bool(groq_client),
        "voice_model_loaded": False,  # Will be True when TTS is added
        "supported_languages": ["darija", "english", "french", "franco-arabic"],
        "timestamp": datetime.now().isoformat()
    }

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
        
        response = ChatResponse(
            response=ai_response,
            detected_language=detected_lang,
            audio_url=None,  # Will add TTS later
            timestamp=datetime.now(),
            personality_mode="superico_streamer"
        )
        
        logger.info(f"🤖 Response: {ai_response}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
