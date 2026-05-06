import os
import shutil
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fingerprinter import init_db, identify
import sqlite3

app = FastAPI(title="Audio Identification System")

DB_PATH = "fingerprints.db"

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM songs")
    songs = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM fingerprints")
    fps = c.fetchone()[0]
    conn.close()
    return {"status": "ok", "songs_indexed": songs, "fingerprints": fps}

@app.post("/identify")
async def identify_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
        raise HTTPException(400, "Unsupported format. Use mp3/wav/ogg/m4a")
    
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        result = identify(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(500, f"Processing error: {str(e)}")
    
    os.unlink(tmp_path)
    
    if result["match"] is None:
        return JSONResponse({
            "status": "no_match",
            "message": "No reliable match found",
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"]
        })
    
    return JSONResponse({
        "status": "match_found",
        "match": result["match"],
        "latency_ms": result["latency_ms"]
    })

@app.get("/songs")
def list_songs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT song_id, title, artist, genre FROM songs LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return {"songs": [{"song_id": r[0], "title": r[1], "artist": r[2], "genre": r[3]} for r in rows]}
