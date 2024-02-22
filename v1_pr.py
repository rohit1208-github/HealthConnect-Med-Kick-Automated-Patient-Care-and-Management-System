import librosa
import noisereduce as nr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio_input, sampling_rate = librosa.load(r"C:\Users\prohi\OneDrive\Desktop\Med-Kick\archive\Medical Speech, Transcription, and Intent\recordings\train\1249120_44246595_62839015.wav", sr=16000)

audio_denoised = nr.reduce_noise(audio_clip=audio_input, sr=sampling_rate)

inputs = processor(audio_denoised, return_tensors="pt", sampling_rate=16000).input_values

logits = model(inputs).logits

ids = torch.argmax(logits, axis=-1)

text_output = processor.decode(ids)[0]
print(text_output)