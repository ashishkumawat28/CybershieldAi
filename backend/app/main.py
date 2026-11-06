
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(
    title="CyberShield AI",
    description="AI-Powered Cybersecurity Threat Detection System",
    version="1.0.0"
)

# CORS configuration for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to CyberShield AI API",
        "project": "CyberShield AI",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "Connected",
        "project": "CyberShield AI",
        "backend": "FastAPI",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/info")
def project_info():
    return {
        "name": "CyberShield AI",
        "purpose": "Intelligent Cybersecurity Threat Detection",
        "modules": [
            "Network Intrusion Detection",
            "Anomaly Detection",
            "Phishing Detection",
            "Attack Classification",
            "AI Security Assistant"
        ]
    }
