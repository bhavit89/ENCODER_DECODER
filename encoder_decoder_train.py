from EncoderDecoderUtils import prepare_data
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm 
from  torch.utils.data import DataLoader,TensorDataset
import nltk
from nltk.translate.bleu_score import sentence_bleu
import warnings 
warnings.filterwarnings("ignore")

DATA_PATH = "/home/bhavit/Desktop/Road_to_transformers/hind-english/hindi_english_parallel.csv"
BATCH_SIZE = 8
MAX_VOCAB_SIZE = 1000
MIN_FREQ = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
train_dataset, test_datset, hindi_vocab, english_vocab = prepare_data(
    DATA_PATH, 
    batch_size=BATCH_SIZE, 
    max_vocab_size=MAX_VOCAB_SIZE, 
    min_freq=MIN_FREQ
)

print("DATA PREPRATION DONE..........")

# ENCODER 

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,dropout):
        super(Encoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=dropout)


    def forward(self,x):
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        outputs ,(hidden,cell) = self.lstm(embedding)


        return hidden,cell
    
# DECODER 
class Decoder(nn.Module):
    def __init__ (self,input_size,embedding_size,hidden_size,output_size,num_layers,dropout):
        super(Decoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=dropout)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden,cell):
        x = x.unsqueeze(0) # Add sequence length dimension
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        output , (hidden,cell) = self.lstm(embedding,(hidden,cell))
        prediction = self.fc(output)
        prediction = prediction.squeeze(0) # Remove sequence length dimension

        return prediction ,hidden,cell
    
# SEQUENCE TO SEQUENCE 
class Seq2Seq(nn.Module):
    def __init__(self, encoder ,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder 

    
    def forward(self, source ,target, teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english_vocab)

        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device=device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    

    
input_size_encoder = len(hindi_vocab)
input_size_decoder = len(english_vocab)
output_size = len(english_vocab)
encoder_embedding_size = 256
decoder_embedding_size = 256
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
learning_rate = 0.001
num_epochs = 150

encoder = Encoder(input_size=input_size_encoder,
                  embedding_size=encoder_embedding_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  dropout=enc_dropout).to(device)

decoder = Decoder(input_size=input_size_decoder,
                  embedding_size=decoder_embedding_size,
                  hidden_size=hidden_size,
                  output_size=output_size,
                  num_layers=num_layers,
                  dropout=dec_dropout).to(device)


print("ARCHITECTURE DEFINATION DONE........")

model =  Seq2Seq(encoder,decoder).to(device)
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=english_vocab["<pad>"])


for epoch in  range(num_epochs):
    print(f"EPOCH {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0

    loop = tqdm(enumerate(train_dataset),total=len(train_dataset),leave=True)
    for batch_idx , (source,target) in loop :
        source = source.to(device).transpose(0,1) # Transpose to sequence_length ,batch_size because LSTM expects that way
        target = target.to(device).transpose(0,1)

        # FORWARD PASS 
        optimizer.zero_grad()
        output = model(source,target)

        # Reshape for loss computation
        # Flatten output: [target_len, batch_size, target_vocab_size] -> [(target_len * batch_size), target_vocab_size]
        # Flatten target: [target_len, batch_size] -> [(target_len * batch_size)]

        output = output[1:].reshape(-1 ,output.shape[2]) # skip <sos> token
        target = target[1:].reshape(-1)

        loss = criterion(output,target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) #  For exploding gradient problems in LSTM
        optimizer.step()

        epoch_loss += loss.item()

        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch Loss: {epoch_loss / len(train_dataset):.4f}")
