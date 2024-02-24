import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import asyncio
import threading

from transcriber import AudioTranscriber

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

event_loop = asyncio.new_event_loop()

transcriber = AudioTranscriber(event_loop, pipe)

def start_transcription():
    global transcriber, event_loop, thread

    asyncio.set_event_loop(event_loop)
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()

    asyncio.run_coroutine_threadsafe(transcriber.start_transcription(2), event_loop)

start_transcription()

while True:
    pass

# y, sr = librosa.load('najs.mp3')

# sample = { "array": y[:300000], "sampling_rate": sr }
# print(y)
# result = pipe(sample, return_timestamps=True)

# for c in result["chunks"]:
#   print(c["timestamp"], ": ", c["text"])