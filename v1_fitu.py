import pandas as pd
import os
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast 
from tqdm import tqdm

data = pd.read_csv(r'C:\Users\prohi\OneDrive\Desktop\Med-Kick\trns.csv')

base_dir = r"C:\Users\prohi\OneDrive\Desktop\Med-Kick\recordings"
train_dir, test_dir = os.path.join(base_dir, "train"), os.path.join(base_dir, "test")

data['file_path'] = data['file_name'].apply(lambda x: os.path.join(train_dir, x) if os.path.exists(os.path.join(train_dir, x)) else (os.path.join(test_dir, x) if os.path.exists(os.path.join(test_dir, x)) else None))
data.dropna(subset=['file_path'], inplace=True)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

batch_size = 4

class AudioDataset(Dataset):
    def __init__(self, dataframe, processor, sample_rate=16000):
        self.dataframe = dataframe
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        speech, _ = librosa.load(row['file_path'], sr=self.sample_rate)
        input_values = self.processor(speech, return_tensors="pt", padding="longest", sampling_rate=self.sample_rate).input_values.squeeze()
        labels = self.processor.tokenizer.encode(row['phrase'], return_tensors="pt").squeeze()
        return {"input_values": input_values, "labels": labels}

def collate_batch(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_values = pad_sequence(input_values, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    
    return {
        'input_values': input_values,
        'labels': labels
    }

train_dataset = AudioDataset(train_data, processor)
test_dataset = AudioDataset(test_data, processor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

scaler = GradScaler()

for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        with autocast():
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()

model.save_pretrained(r"C:\Users\prohi\OneDrive\Desktop\Med-Kick\model")
processor.save_pretrained(r"C:\Users\prohi\OneDrive\Desktop\Med-Kick\model")