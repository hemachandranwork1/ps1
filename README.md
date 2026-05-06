# Audio Identification & Source Detection System

## Team Information
- **Team Name**: unknown
- **Year**: 2026
- **All-Female Team**:  NO

## Architecture Overview


**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
- We extract spectrogram peaks using librosa STFT, generate combinatorial hashes from peak pairs with time deltas, and store them in SQLite with hash indexing for O(1) lookup.
- We use Shazam-style landmark fingerprinting: query hashes are matched against the database, and the song with the highest time-offset alignment score wins.
- SQLite hash index handles thousands of songs; FastAPI async endpoints handle concurrent queries natively.
- Log-scaled spectrograms and peak-based hashing are inherently noise-robust; a confidence threshold filters false positives.
