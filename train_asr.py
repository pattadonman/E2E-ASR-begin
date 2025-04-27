# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
# import torchaudio
# from torch.utils.data import DataLoader
# from datasets import load_dataset

# # Load pre-trained Whisper model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-large")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# # Load the dataset (ตัวอย่างใช้ CommonVoice ภาษาไทย หรือสามารถเลือก dataset อื่น ๆ ได้)
# dataset = load_dataset("mozilla-foundation/common_voice_11_0", "th", split="train[:1%]", trust_remote_code=True)
# dataset = dataset.map(lambda x: processor(x["audio"]["array"], sampling_rate=16000, return_tensors="pt"))

# # Preprocessing function (ปรับแต่งข้อมูลเสียง)
# def preprocess_audio(batch):
#     # ตรวจสอบว่า batch["audio"] เป็น dictionary หรือไม่
#     audio_input = batch["audio"]
    
#     # แปลงข้อมูลให้เป็น numpy array ถ้าจำเป็น
#     if isinstance(audio_input, dict):
#         audio_input = audio_input["array"]  # หรือบางกรณีอาจจะเป็น "path" หรือ "file" ขึ้นอยู่กับโครงสร้าง

#     # ตรวจสอบว่า audio_input เป็น numpy array หรือไม่
#     if isinstance(audio_input, np.ndarray):
#         # สามารถส่งเข้า processor ได้
#         return processor(audio_input, sampling_rate=16000, return_tensors="pt")
#     else:
#         raise ValueError("audio_input ต้องเป็น numpy array ที่มีข้อมูลเสียง")

# # Define DataLoader
# train_dataloader = DataLoader(dataset, batch_size=4, collate_fn=preprocess_audio)

# # Training loop
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# # Number of epochs
# epochs = 3

# for epoch in range(epochs):
#     model.train()
#     for batch in train_dataloader:
#         inputs = batch["input_values"].squeeze(1).to("cuda" if torch.cuda.is_available() else "cpu")
#         labels = batch["labels"].squeeze(1).to("cuda" if torch.cuda.is_available() else "cpu")
        
#         optimizer.zero_grad()
#         outputs = model(input_values=inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
    
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# # Save the trained model
# model.save_pretrained("./trained_whisper_thai_model")














import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torchaudio
from torch.utils.data import DataLoader
import numpy as np

# Load pre-trained Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# Load the dataset (ตัวอย่างใช้ CommonVoice ภาษาไทย หรือสามารถเลือก dataset อื่น ๆ ได้)
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "th", split="train[:1%]", trust_remote_code=True)

# Preprocessing function (ปรับแต่งข้อมูลเสียง)
def preprocess_audio(batch):
    # ตรวจสอบว่า batch["audio"] เป็น dictionary หรือไม่
    audio_input = batch["audio"]
    
    # แปลงข้อมูลให้เป็น numpy array ถ้าจำเป็น
    if isinstance(audio_input, dict):
        audio_input = audio_input["array"]  # หรือบางกรณีอาจจะเป็น "path" หรือ "file" ขึ้นอยู่กับโครงสร้าง

    # ตรวจสอบว่า audio_input เป็น numpy array หรือไม่
    if isinstance(audio_input, np.ndarray):
        # สามารถส่งเข้า processor ได้
        processed = processor(audio_input, sampling_rate=16000, return_tensors="pt")
        # ตรวจสอบว่า key "input_values" ถูกสร้าง
        if "input_values" not in processed:
            raise ValueError("Missing 'input_values' in the processed data")
        return processed
    else:
        raise ValueError("audio_input ต้องเป็น numpy array ที่มีข้อมูลเสียง")

# Apply preprocessing to the dataset
dataset = dataset.map(lambda x: preprocess_audio(x), remove_columns=["audio"])

# ตรวจสอบข้อมูลใน dataset
print(dataset[0])  # พิมพ์ข้อมูลแรกเพื่อให้แน่ใจว่า "input_values" มีอยู่

# Define DataLoader
train_dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda batch: {
    "input_values": torch.stack([x["input_values"] for x in batch]),
    "labels": torch.stack([x["labels"] for x in batch])
})

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Number of epochs
epochs = 3

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        inputs = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
model.save_pretrained("./trained_whisper_thai_model")
