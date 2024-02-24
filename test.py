import librosa
import soundfile as sf
from utils import get_valid_input_devices

def get_valid_devices():
    devices = get_valid_input_devices()
    return [
        {
            "index": d["index"],
            "name": f"{d['name']} {d['host_api_name']} ({d['max_input_channels']} in)",
        }
        for d in devices
    ]

print(get_valid_devices())

# y, sr = librosa.load('najs.mp3')
# audio = y[:300000]
# sf.write("x.wav", audio, sr)