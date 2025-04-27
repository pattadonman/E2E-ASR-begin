from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch

# Load the fine-tuned model
model = WhisperForConditionalGeneration.from_pretrained("./trained_whisper_thai_model")
processor = WhisperProcessor.from_pretrained("./trained_whisper_thai_model")

# Inference function
def transcribe_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_values"].squeeze(0).to("cuda" if torch.cuda.is_available() else "cpu"))
    
    transcription = processor.decode(generated_ids[0], skip_special_tokens=True)
    return transcription

# Test the transcription
file_path = "path_to_audio_file.wav"  # แก้ไขเป็นพาธของไฟล์เสียงที่คุณต้องการทดสอบ
transcription = transcribe_audio(file_path)
print(f"Transcription: {transcription}")
