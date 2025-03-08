import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MeldDataset(Dataset):
  def __init__(self, csv_path, video_dir):
    # Ensure the file exists before reading
    if not os.path.exists(csv_path):
      raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")
    
    self.data = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully with {len(self.data)} records.")
    
    self.video_dir = video_dir
    self.tokenizer = AutoTokenizer.from_pretrained("berk-base-uncased")
    
    self.emotion_map = {
      "anger":0,
      "disgust":1,
      "fear":2,
      "joy":3,
      "sadness":4,
      "surprise":5
    }
    
    self.sentiment_map = {
      "negative":0,
      "neutral":1,
      "positive":2
    }
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    video_filename = f""""""

if __name__ == '__main__':
  # Get the absolute path of the dataset directory
  base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
  csv_path = os.path.join(base_dir, "..", "dataset", "dev", "dev_sent_emo.csv")
  video_dir = os.path.join(base_dir, "..", "dataset", "dev", "dev_splits_complete")

  # Debugging: Print paths to verify correctness
  print(f"CSV Path: {csv_path}")
  print(f"Video Directory: {video_dir}")

  # Create dataset instance
  try:
    meld = MeldDataset(csv_path, video_dir)
  except FileNotFoundError as e:
    print(e)
