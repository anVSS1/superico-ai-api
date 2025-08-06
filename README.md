# 🎮 Superico AI Streamer

Multilingual AI clone of Twitch streamer Superico with voice cloning, supporting Darija, English, French, and Franco-Arabic.

## 🚀 Quick Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

## 📋 Features

- **🎤 Voice Cloning:** XTTS v2 trained on Superico's voice
- **🌍 Multilingual:** Darija, English, French, Franco-Arabic
- **🤖 AI Personality:** LLaMA-3 via Groq API  
- **🎮 OBS Integration:** Real-time audio for streaming
- **☁️ Free Hosting:** Deploy on Render.com/Railway ($0/month)

## 🏗️ Project Structure

```
cloud_api/                 # Deploy this folder to cloud
├── main.py                # FastAPI server
├── requirements.txt       # Python dependencies  
├── Dockerfile            # Container config
└── reference_audio.wav   # Voice model (add your own)
```

## ⚡ Deploy to Cloud

### 1. Environment Variables Required:
```
GROQ_API_KEY=your_groq_api_key_here
PORT=8000
```

### 2. Get Groq API Key (Free):
1. Go to [console.groq.com](https://console.groq.com)
2. Create account
3. Copy API key from dashboard

### 3. Deploy on Render.com:
1. Connect this GitHub repo
2. Choose "Web Service" 
3. Set Environment: **Docker**
4. Add environment variables above
5. Deploy!

### 4. Deploy on Railway.app:
1. New Project → Deploy from GitHub
2. Add environment variables
3. Deploy!

## 🎮 Usage

### For Streamers:
1. Deploy API (get your URL like `https://your-app.onrender.com`)
2. Download OBS integration script 
3. Run locally alongside OBS
4. Viewers use `!q <question>` in chat
5. AI responds with cloned voice!

### Test Commands:
- `!q wash kayn Superico?` (Darija)
- `!q how do you play this game?` (English)  
- `!q salut comment ça va?` (French)
- `!q yo bro wash ready?` (Franco-Arabic)

## 📱 API Endpoints

- `GET /` - Health check
- `POST /chat` - Process chat message
- `GET /health` - Detailed status

### Chat API Example:
```bash
curl -X POST "https://your-app.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "wash kayn Superico?",
    "username": "viewer123"
  }'
```

## 🔧 Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 💰 Cost: $0/month

- Render.com free tier
- Groq API free tier  
- Railway.app free tier

## 🎯 Live Demo

Once deployed, test at: `https://your-app-url.com/`

---

**Ready to create your multilingual AI streaming clone!** 🎮
