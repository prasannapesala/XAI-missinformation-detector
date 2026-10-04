import torch
from transformers import RobertaTokenizer
import shap
import spacy
import streamlit

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print("RoBERTa tokenizer: OK")

nlp = spacy.load("en_core_web_sm")
print("spaCy: OK")

print("\n All good! Ready to start coding.")    