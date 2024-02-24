from concurrent.futures import ThreadPoolExecutor
import functools
import queue
import asyncio
import numpy as np

from vad import Vad
from utils import create_audio_stream

#from gpt import OpenAIAPI
#api = OpenAIAPI()

class AudioTranscriber:
    def __init__(self, event_loop, model, silence_limit=8, noise_threshold=5, history_limit=5, speech_limit=200, language=None):
        self.transcribing = False
        self.event_loop = event_loop
        self.audio_queue = queue.Queue()
        self.model = model
        self._running = asyncio.Event()
        self.vad = Vad(threshold=0.8)
        self.silence_counter: int = 0
        self.audio_data_list = []
        self._transcribe_task = None
        self.history = []

        self.speech_limit = speech_limit
        self.silence_limit = silence_limit
        self.noise_threshold = noise_threshold
        self.language = language
        self.history_limit = history_limit
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def transcribe(self):
        while self.transcribing:
            try:
                # Get raw audio from queue with 3 sec timoue
                audio_data = await self.event_loop.run_in_executor(
                    self.executor, functools.partial(self.audio_queue.get, timeout=3.0)
                )

                print("-")

                inference = functools.partial(
                    self.model.transcribe,
                    audio=audio_data,
                    language=self.language
                )

                segments, _  = await self.event_loop.run_in_executor(self.executor, inference)

                for segment in segments:
                    #text = await api.proofread(segment.text, self.history)
                    text = segment.text
                    print(">", text)
                    self.history.append(text)

                    if len(self.history) > self.history_limit:
                        self.history = self.history[1:]

            except queue.Empty:
                # Queue get timed out, skip to next
                continue
            except Exception as e:
                print("Transcribe error", e)

    def process_audio(self, audio_data: np.ndarray, frames: int, time, status):
        is_speech = self.vad.is_speech(audio_data, self.stream.samplerate)
        if is_speech:
            self.silence_counter = 0
            self.audio_data_list.append(audio_data.flatten())
        else:
            self.silence_counter += 1

        limit_reached = len(self.audio_data_list) > self.speech_limit

        if not is_speech:
            speech_trigger = len(self.audio_data_list) > self.speech_limit
            silence_trigger = self.silence_counter > self.silence_limit

            if speech_trigger or silence_trigger:
                self.silence_counter = 0

                if len(self.audio_data_list) > self.noise_threshold:
                    concatenate_audio_data = np.concatenate(self.audio_data_list)
                    self.audio_queue.put(concatenate_audio_data)

                self.audio_data_list.clear()

    async def start_transcription(self, audio_device):
        try:
            print("Starting transcription")
            self.transcribing = True
            self.stream = create_audio_stream(audio_device, self.process_audio)
            self.stream.start()
            self._running.set()
            self._transcribe_task = asyncio.run_coroutine_threadsafe(
                self.transcribe(), self.event_loop
            )
            while self._running.is_set():
                await asyncio.sleep(1)

        except Exception as e:
            print("Could not start transcription:", e)

    async def stop_transcription(self):
        try:
            print("Stopping transcription")
            self.transcribing = False
            if self._transcribe_task is not None:
                self.event_loop.call_soon_threadsafe(self._transcribe_task.cancel)
                self._transcribe_task = None

            if self.stream is not None:
                self._running.clear()
                self.stream.stop()
                self.stream.close()
                self.stream = None

        except Exception as e:
            print("Could not stop transcription:", e)