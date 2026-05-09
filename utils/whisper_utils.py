import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

def record_audio(duration=30, samplerate=16000):
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        print("Recording done.")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, samplerate, audio)
        return tmp.name
    except Exception as e:
        print(f"Recording error: {e}")
        return None

def transcribe(filepath: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Voice transcription unavailable (no OpenAI key). Please use text input."
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        with open(filepath, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return result.text
    except Exception as e:
        return f"Transcription failed: {str(e)}"

def transcribe_audio(filepath: str) -> str:
    return transcribe(filepath)

def count_filler_words(text: str) -> dict:
    fillers = ["um", "uh", "like", "you know", "basically",
               "literally", "actually", "so", "right"]
    text_lower = text.lower()
    counts = {}
    for word in fillers:
        count = text_lower.split().count(word)
        if count > 0:
            counts[word] = count
    return counts