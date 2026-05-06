import os
import shutil
import tempfile
import time
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fingerprinter import init_db, identify, get_spectrogram, load_audio, get_peaks, generate_hashes
import numpy as np

app = FastAPI(title="Audio Identification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DB_PATH = "fingerprints.db"

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def root():
    with open("ui.html") as f:
        return f.read()

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

@app.get("/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT genre, COUNT(*) FROM songs GROUP BY genre")
    genres = dict(c.fetchall())
    c.execute("SELECT COUNT(*) FROM fingerprints")
    fps = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM songs")
    songs = c.fetchone()[0]
    conn.close()
    return {"genres": genres, "total_songs": songs, "total_fingerprints": fps}

@app.get("/songs")
def list_songs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT song_id, title, artist, genre, duration FROM songs")
    rows = c.fetchall()
    conn.close()
    return {"songs": [{"song_id": r[0], "title": r[1], "artist": r[2], "genre": r[3], "duration": r[4]} for r in rows]}

def identify_top3(filepath):
    start = time.time()
    from fingerprinter import fingerprint_audio, CONFIDENCE_THRESHOLD
    hashes = fingerprint_audio(filepath)
    if not hashes:
        return {"match": None, "top3": [], "confidence": 0, "latency_ms": 0}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    matches = {}
    for h, query_offset in hashes:
        c.execute("SELECT song_id, offset FROM fingerprints WHERE hash=?", (h,))
        rows = c.fetchall()
        for song_id, db_offset in rows:
            delta = db_offset - query_offset
            key = (song_id, delta)
            matches[key] = matches.get(key, 0) + 1

    if not matches:
        conn.close()
        return {"match": None, "top3": [], "confidence": 0, "latency_ms": round((time.time()-start)*1000,2)}

    song_scores = {}
    for (song_id, delta), count in matches.items():
        if song_id not in song_scores or song_scores[song_id] < count:
            song_scores[song_id] = count

    sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    total_hashes = len(hashes)

    top3 = []
    for song_id, count in sorted_songs:
        c.execute("SELECT song_id,title,artist,duration,genre FROM songs WHERE song_id=?", (song_id,))
        row = c.fetchone()
        if row:
            top3.append({
                "song_id": row[0], "title": row[1], "artist": row[2],
                "duration": row[3], "genre": row[4],
                "confidence": round(count / total_hashes, 4)
            })
    conn.close()

    latency = round((time.time()-start)*1000, 2)
    best = top3[0] if top3 and top3[0]["confidence"] >= CONFIDENCE_THRESHOLD else None
    return {"match": best, "top3": top3, "confidence": top3[0]["confidence"] if top3 else 0, "latency_ms": latency}

@app.post("/identify")
async def identify_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = identify_top3(tmp_path)
        y, sr = load_audio(tmp_path)
        D = get_spectrogram(y, sr)
        peaks = get_peaks(D)
        peak_data = [(int(f), int(t)) for f, t, _ in peaks[:80]]
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(500, f"Processing error: {str(e)}")
    os.unlink(tmp_path)

    return JSONResponse({
        "status": "match_found" if result["match"] else "no_match",
        "match": result["match"],
        "top3": result["top3"],
        "latency_ms": result["latency_ms"],
        "peaks": peak_data
    })

@app.post("/identify/batch")
async def identify_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        try:
            result = identify_top3(tmp_path)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            os.unlink(tmp_path)
    return {"results": results, "total": len(results)}
