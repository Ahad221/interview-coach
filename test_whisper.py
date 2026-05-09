from utils.whisper_utils import record_audio, transcribe_audio, count_filler_words

print("Testing microphone — speak for 5 seconds...")
path = record_audio(duration=5)
print(f"Saved to: {path}")

text = transcribe_audio(path)
print(f"Transcript: {text}")

fillers = count_filler_words(text)
print(f"Filler words: {fillers}")