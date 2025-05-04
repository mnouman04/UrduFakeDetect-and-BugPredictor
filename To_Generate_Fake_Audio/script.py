import os
import random
from gtts import gTTS
from pathlib import Path

# Urdu sample phrases
urdu_sentences = [
    "میرا نام علی ہے۔",
    "آج موسم بہت خوشگوار ہے۔",
    "میں یونیورسٹی جا رہا ہوں۔",
    "یہ ایک خوبصورت دن ہے۔",
    "مجھے چائے پسند ہے۔",
    "کیا آپ میری مدد کریں گے؟",
    "آج کا لیکچر بہت دلچسپ تھا۔",
    "مجھے نئی چیزیں سیکھنے کا شوق ہے۔",
    "کل ہم پارک جائیں گے۔",
    "آپ کا شکریہ۔"
]

# Directory setup
output_dir = Path("generated_audio")
bonafide_dir = output_dir / "bonafide"
deepfake_dir = output_dir / "deepfake"
bonafide_dir.mkdir(parents=True, exist_ok=True)
deepfake_dir.mkdir(parents=True, exist_ok=True)

# Function to generate audio
def generate_audio(label="bonafide", count=5):
    for i in range(count):
        sentence = random.choice(urdu_sentences)
        tts = gTTS(text=sentence, lang='ur')
        filename = f"{label}_{i+1}.mp3"
        filepath = (bonafide_dir if label == "bonafide" else deepfake_dir) / filename
        tts.save(filepath)
        print(f"Generated {label} audio: {filepath}")

# Generate 5 bonafide and 5 deepfake samples
generate_audio(label="bonafide", count=5)
generate_audio(label="deepfake", count=5)
