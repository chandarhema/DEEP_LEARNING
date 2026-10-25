# question_2_gru_seq2seq.py
import os,re,time,random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH="english_hindi.csv"
ENGLISH_COL="english_sentence"
HINDI_COL="hindi_sentence"
MAX_VOCAB=12000
MAX_LEN=20
BATCH_SIZE=64
EPOCHS=10
EMBED_DIM=128
HIDDEN_DIM=256
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42);np.random.seed(42);torch.manual_seed(42)

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]",str(text).lower(),re.UNICODE)

df=pd.read_csv(DATA_PATH).dropna(subset=[ENGLISH_COL,HINDI_COL])
df=df[[ENGLISH_COL,HINDI_COL]].astype(str).drop_duplicates()
df=df[(df[ENGLISH_COL].str.strip()!="")&(df[HINDI_COL].str.strip()!="")]
print("Dataset rows:",len(df),"Columns:",df.columns.tolist())

def build_vocab(texts):
    from collections import Counter
    counts=Counter(t for s in texts for t in tokenize(s))
    vocab={"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}
    for word,_ in counts.most_common(MAX_VOCAB-4):vocab[word]=len(vocab)
    return vocab

en_vocab=build_vocab(df[ENGLISH_COL].tolist())
hi_vocab=build_vocab(df[HINDI_COL].tolist())

def encode(text,vocab):
    ids=[vocab.get(w,3) for w in tokenize(text)[:MAX_LEN-2]]
    return [vocab["<SOS>"]]+ids+[vocab["<EOS>"]]

def pad(ids):
    return ids[:MAX_LEN]+[0]*max(0,MAX_LEN-len(ids))

class TranslationDataset(Dataset):
    def __init__(self,data):
        self.en=[pad(encode(x,en_vocab)) for x in data[ENGLISH_COL]]
        self.hi=[pad(encode(x,hi_vocab)) for x in data[HINDI_COL]]
    def __len__(self):return len(self.en)
    def __getitem__(self,i):return torch.tensor(self.en[i]),torch.tensor(self.hi[i])

train_df,test_df=train_test_split(df,test_size=0.1,random_state=42)
train_df,val_df=train_test_split(train_df,test_size=0.1,random_state=42)
train_loader=DataLoader(TranslationDataset(train_df),batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(TranslationDataset(val_df),batch_size=BATCH_SIZE)
test_loader=DataLoader(TranslationDataset(test_df),batch_size=BATCH_SIZE)

class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,EMBED_DIM,padding_idx=0)
        self.gru=nn.GRU(EMBED_DIM,HIDDEN_DIM,batch_first=True)
    def forward(self,x):
        _,h=self.gru(self.embedding(x))
        return h

class Decoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,EMBED_DIM,padding_idx=0)
        self.gru=nn.GRU(EMBED_DIM,HIDDEN_DIM,batch_first=True)
        self.fc=nn.Linear(HIDDEN_DIM,vocab_size)
    def forward(self,x,h):
        out,h=self.gru(self.embedding(x),h)
        return self.fc(out),h

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder(len(en_vocab))
        self.decoder=Decoder(len(hi_vocab))
    def forward(self,src,tgt):
        h=self.encoder(src)
        logits,_=self.decoder(tgt[:,:-1],h)
        return logits

model=Seq2Seq().to(DEVICE)
criterion=nn.CrossEntropyLoss(ignore_index=0)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

def evaluate(loader):
    model.eval();total=0;tokens=0
    with torch.no_grad():
        for src,tgt in loader:
            src,tgt=src.to(DEVICE),tgt.to(DEVICE)
            logits=model(src,tgt)
            loss=criterion(logits.reshape(-1,len(hi_vocab)),tgt[:,1:].reshape(-1))
            n=(tgt[:,1:]!=0).sum().item()
            total+=loss.item()*n;tokens+=n
    return total/max(tokens,1)

start=time.time()
for epoch in range(1,EPOCHS+1):
    model.train();total=0;tokens=0
    for src,tgt in train_loader:
        src,tgt=src.to(DEVICE),tgt.to(DEVICE)
        optimizer.zero_grad()
        logits=model(src,tgt)
        loss=criterion(logits.reshape(-1,len(hi_vocab)),tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        n=(tgt[:,1:]!=0).sum().item();total+=loss.item()*n;tokens+=n
    train_loss=total/max(tokens,1);val_loss=evaluate(val_loader)
    print(f"Epoch {epoch}/{EPOCHS} | train loss={train_loss:.4f} | validation loss={val_loss:.4f}")
test_loss=evaluate(test_loader)
print(f"Test loss: {test_loss:.4f} | Training time: {time.time()-start:.1f} seconds")

id_to_hi={i:w for w,i in hi_vocab.items()}
def translate(sentence):
    model.eval()
    src=torch.tensor([pad(encode(sentence,en_vocab))],device=DEVICE)
    with torch.no_grad():
        h=model.encoder(src)
        token=torch.tensor([[hi_vocab["<SOS>"]]],device=DEVICE)
        result=[]
        for _ in range(MAX_LEN-1):
            logits,h=model.decoder(token,h)
            next_id=logits[:,-1,:].argmax(-1).item()
            if next_id in (hi_vocab["<EOS>"],hi_vocab["<PAD>"]):break
            if next_id!=hi_vocab["<SOS>"]:result.append(id_to_hi.get(next_id,"<UNK>"))
            token=torch.tensor([[next_id]],device=DEVICE)
    return " ".join(result)

torch.save({"model":model.state_dict(),"en_vocab":en_vocab,"hi_vocab":hi_vocab},"gru_seq2seq.pt")
print("Saved model: gru_seq2seq.pt")
while True:
    sentence=input("English sentence (or 'quit'): ").strip()
    if sentence.lower()=="quit":break
    print("Hindi translation:",translate(sentence))