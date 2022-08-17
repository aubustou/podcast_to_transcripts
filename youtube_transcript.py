import json
import os
from pathlib import Path
from asrecognition import ASREngine
import sys
import subprocess

video = sys.argv[1]

INTERMEDIATE_FILE = "out.wav"

subprocess.run(["yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--output", INTERMEDIATE_FILE,
                f"https://www.youtube.com/watch?v={video}"])

subprocess.run(["ffmpeg",
                "-i", INTERMEDIATE_FILE,
                "-f", "segment",
                "-segment_time", "30",
                "-c", "copy",
                "out.%0d.wav"])


# MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
MODEL = "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french"

SPLITTED_FILE = "out.*.wav"

asr = ASREngine("fr", device="cuda", model_path=MODEL)

audio_paths = sorted([str(x) for x in Path().glob(SPLITTED_FILE)])

transcriptions = asr.transcribe(audio_paths)
print(transcriptions)
json.dump(transcriptions, Path(f"out-{video}-{MODEL.split('/')[-1]}.json").open("w"), indent=4)

try:
    os.unlink(INTERMEDIATE_FILE)
    for path in audio_paths:
        os.unlink(path)
except ValueError:
    pass
