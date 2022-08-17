import json
import os
from pathlib import Path
from asrecognition import ASREngine
import sys
import subprocess


#MODEL = "facebook/s2t-small-mustc-en-fr-st"
MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
# MODEL = "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french"
# MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
TEMPORARY_WAV_FILE = "output.wav"

audio_paths: list[Path] = []

input_file = sys.argv[1]

try:
    subprocess.run(["ffmpeg",
                    "-i", input_file,
                    "-acodec", "pcm_u8",
                    "-ar", "22050",
                    TEMPORARY_WAV_FILE])

    subprocess.run(["ffmpeg",
                    "-i", TEMPORARY_WAV_FILE,
                    "-f", "segment",
                    "-segment_time", "30",
                    "-c", "copy",
                    "out.%0d.wav"])


    asr = ASREngine("fr", device="cuda", model_path=MODEL)

    SPLITTED_FILE = "out.*.wav"
    audio_paths = sorted([str(x) for x in Path().glob(SPLITTED_FILE)])

    transcriptions = asr.transcribe(audio_paths)
    print(transcriptions)
    json.dump(transcriptions, Path(f"out-{input_file}-{MODEL.split('/')[-1]}.json").open("w"), indent=4)


finally:
    try:
        for path in audio_paths:
            os.unlink(path)
        os.unlink(TEMPORARY_WAV_FILE)
    except ValueError:
        pass
