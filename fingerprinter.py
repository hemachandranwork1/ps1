import os
import pickle
import hashlib
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import sqlite3
import time

DB_PATH = "fingerprints.db"
SAMPLE_RATE = 8000
N_FFT = 4096
HOP_LENGTH = 512
PEAK_NEIGHBORHOOD = 20
FAN_VALUE = 15
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
CONFIDENCE_THRESHOLD = 0.01

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS songs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  song_id TEXT, title TEXT, artist TEXT,
                  duration REAL, genre TEXT, filepath TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS fingerprints
                 (hash TEXT, song_id TEXT, offset REAL)''')
    c.execute('CREATE INDEX IF NOT EXISTS hash_idx ON fingerprints(hash)')
    conn.commit()
    conn.close()

def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return y, sr

def get_spectrogram(y, sr):
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    D = librosa.amplitude_to_db(D, ref=np.max)
    return D

def get_peaks(D):
    struct = generate_binary_structure(2, 1)
    neighborhood = maximum_filter(D, size=PEAK_NEIGHBORHOOD, mode='constant')
    local_max = (D == neighborhood)
    eroded = binary_erosion(D == D.max(), structure=struct, border_value=1)
    detected_peaks = local_max ^ eroded
    amps = D[detected_peaks]
    freqs, times = np.where(detected_peaks)
    peaks = list(zip(freqs, times, amps))
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:200]
    return peaks

def generate_hashes(peaks):
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE):
            if i + j < len(peaks):
                f1 = peaks[i][0]
                f2 = peaks[i + j][0]
                t1 = peaks[i][1]
                t2 = peaks[i + j][1]
                dt = t2 - t1
                if MIN_HASH_TIME_DELTA <= dt <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(f"{f1}|{f2}|{dt}".encode()).hexdigest()[:20]
                    hashes.append((h, t1))
    return hashes

def fingerprint_audio(filepath):
    y, sr = load_audio(filepath)
    D = get_spectrogram(y, sr)
    peaks = get_peaks(D)
    hashes = generate_hashes(peaks)
    return hashes

def index_song(filepath, song_id, title, artist, duration, genre):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM songs WHERE song_id=?", (song_id,))
    if c.fetchone():
        conn.close()
        return
    c.execute("INSERT INTO songs (song_id,title,artist,duration,genre,filepath) VALUES (?,?,?,?,?,?)",
              (song_id, title, artist, duration, genre, filepath))
    conn.commit()
    hashes = fingerprint_audio(filepath)
    c.executemany("INSERT INTO fingerprints (hash,song_id,offset) VALUES (?,?,?)",
                  [(h, song_id, float(t)) for h, t in hashes])
    conn.commit()
    conn.close()
    print(f"Indexed: {title} ({len(hashes)} hashes)")

def identify(filepath):
    start = time.time()
    hashes = fingerprint_audio(filepath)
    if not hashes:
        return {"match": None, "confidence": 0, "latency_ms": 0}

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

    conn.close()

    if not matches:
        return {"match": None, "confidence": 0.0, "latency_ms": round((time.time()-start)*1000, 2)}

    best_key = max(matches, key=matches.get)
    best_song_id = best_key[0]
    best_count = matches[best_key]
    confidence = round(best_count / len(hashes), 4)

    if confidence < CONFIDENCE_THRESHOLD:
        return {"match": None, "confidence": confidence, "latency_ms": round((time.time()-start)*1000, 2)}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT song_id,title,artist,duration,genre FROM songs WHERE song_id=?", (best_song_id,))
    row = c.fetchone()
    conn.close()

    latency = round((time.time() - start) * 1000, 2)
    if row:
        return {
            "match": {
                "song_id": row[0], "title": row[1], "artist": row[2],
                "duration": row[3], "genre": row[4], "confidence": confidence
            },
            "latency_ms": latency
        }
    return {"match": None, "confidence": 0.0, "latency_ms": latency}
