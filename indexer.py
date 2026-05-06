import os
import sys
from fingerprinter import init_db, index_song
from tqdm import tqdm

def index_dataset(data_dir="data/raw/data"):
    init_db()
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.wav') or f.endswith('.mp3'):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} audio files")
    
    for filepath in tqdm(files):
        fname = os.path.basename(filepath)
        song_id = os.path.splitext(fname)[0]
        genre = os.path.basename(os.path.dirname(filepath))
        parts = song_id.split('.')
        title = song_id
        artist = genre.capitalize()
        try:
            import librosa
            duration = librosa.get_duration(path=filepath)
        except:
            duration = 0.0
        try:
            index_song(filepath, song_id, title, artist, duration, genre)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data"
    index_dataset(data_dir)
    print("Done indexing.")
