import asyncio
import threading
import sys

from faster_whisper import WhisperModel

from transcriber import AudioTranscriber
from utils import get_valid_input_devices

MODEL           = "small"
LANGUAGE        = None
SILENCE_LIMIT   = 8
NOISE_THRESH    = 5

model = WhisperModel(MODEL, compute_type="float32")

event_loop = asyncio.new_event_loop()

transcriber = AudioTranscriber(
    model=model,
    event_loop=event_loop, 
    silence_limit=SILENCE_LIMIT,
    noise_threshold=NOISE_THRESH,
    language=LANGUAGE
)

def start_transcription(audio_device):
    global transcriber, model, event_loop

    asyncio.set_event_loop(event_loop)
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()

    asyncio.run_coroutine_threadsafe(transcriber.start_transcription(audio_device), event_loop)

    return thread

async def main():
    global transcriber, thread

    try:
        devices = get_valid_input_devices()

        print("Select input device index:")

        for d in devices:
            print(f'{d["index"]}. {d["name"]}')

        device = input('Device:')

        thread = start_transcription(int(device))

        thread.join()

    except KeyboardInterrupt:
        print("Exiting...")
        await transcriber.stop_transcription()
        sys.exit(0)

if __name__ == "__main__":
    devices = get_valid_input_devices()

    print("Select input device index:")

    for d in devices:
        print(f'{d["index"]}. {d["name"]}')

    device = input('Device:')

    thread = start_transcription(int(device))

    thread.join()