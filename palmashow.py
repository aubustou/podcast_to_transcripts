import subprocess
import os
import logging
from pathlib import Path
from asrecognition import ASREngine

logging.basicConfig(level=logging.INFO)

TEMPORARY_WAV_FILE = "output.wav"

# MODEL = "facebook/s2t-small-mustc-en-fr-st"
MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
# MODEL = "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french"

logging.info(f"Using model {MODEL}")
asr = ASREngine("fr", device="cuda", model_path=MODEL)

founds = []
for file_ in Path("palmashow").glob("*.mp3"):
    audio_paths = []
    try:
        logging.info(f"Converting {file_}")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(file_),
                "-acodec",
                "pcm_u8",
                "-ar",
                "22050",
                TEMPORARY_WAV_FILE,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                TEMPORARY_WAV_FILE,
                "-f",
                "segment",
                "-segment_time",
                "30",
                "-c",
                "copy",
                "out.%0d.wav",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logging.info(f"Transcribing {file_}")
        SPLITTED_FILE = "out.*.wav"
        audio_paths = sorted([str(x) for x in Path().glob(SPLITTED_FILE)])

        transcriptions = asr.transcribe(audio_paths)
        transcription = transcriptions[0]["transcription"]
        if "petit" in transcription:
            founds.append(file_)
    finally:
        try:
            for path in audio_paths:
                os.unlink(path)
            os.unlink(TEMPORARY_WAV_FILE)
        except ValueError:
            pass

logging.info(f"Found {len(founds)}")
logging.info(founds)
