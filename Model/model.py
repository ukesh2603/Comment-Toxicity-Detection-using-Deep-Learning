from torch import nn


class TextLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_classes):
        super(TextLSTM,self).__init__()
        self.embeddings=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm=nn.LSTM(embedding_dim,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    
    def forward(self,X):
        X=self.embeddings(X)
        output,(hidden,cell_state)=self.lstm(X)
        hidden=hidden.squeeze(0)
        output=self.fc(hidden)
        return output
    
