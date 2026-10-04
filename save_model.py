import sys, os
sys.path.insert(0, '.')

from src.detection.roberta_classifier import train_model

print("Training model...")
model, tokenizer = train_model()

save_path = "D:/roberta_saved"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model saved to D:/roberta_saved")    