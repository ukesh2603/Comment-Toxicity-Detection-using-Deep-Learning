import contractions
import pandas as pd
import torch
import re 
import pickle
from sklearn.model_selection import train_test_split
from torch import nn,optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



with open(r"word2idx.pkl",'rb') as file:
    word2idx=pickle.load(file)



max_len=200

labels=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]


def clean_text(text):
    text=str(text).lower()
    text=contractions.fix(text)
    text=re.sub(r"<.*?>","",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r"\d+","",text)
    text=re.sub(r"https?://\S+|www\.\S+","",text)
    text=re.sub(r"\S+@\S+","",text)
    text=re.sub(r"@\w+","",text)
    text=re.sub(r"#\w+","",text)
    text=re.sub(r"\s+"," ",text)
    text=re.sub(r"[^a-zA-Z0-9\s]","",text)
    return text.strip()

def encode_text(text):
    sentence=clean_text(text)
    
    encode=[]

    for word in sentence.split():
        if word in word2idx:
            encode.append(word2idx[word])
        else:
            encode.append(word2idx["<UNK>"])

    pad_result=[]


    if len(encode)>max_len:
        encode= encode[:max_len]
    else:
        encode=encode+[0]*(max_len-len(encode))
    pad_result.append(encode)
    
    return torch.tensor(pad_result,dtype=torch.long)


def predict_comments(comment,model):
    
    device=next(model.parameters()).device
    sample=encode_text(comment).to(device)

    model.eval()
    with torch.no_grad():
        output = model(sample)
        probability = torch.sigmoid(output)
        prediction = (probability > 0.5).int()
        
    result = {}
    for label, pred, prob in zip(labels, prediction[0], probability[0]):
        result[label] = {
            "prediction": "Yes" if pred.item() else "No",
            "probability": round(prob.item() * 100, 2)
        }
    return result
    