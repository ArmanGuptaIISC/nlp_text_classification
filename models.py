import torch 
import torch.nn as nn

## Model0 - Basic DAN kind of structure
class Model0(nn.Module):
    def __init__(self,pretrained_embedding , in_feat , hidden_size , word2int):
        super(Model0,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=word2int['<PAD>'])
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(in_feat * 2 , hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.Dropout(0.2),
            nn.ELU()
        )
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.ELU()
        )
        self.linear_relu_stack3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.ELU()
        )
        self.linear_relu_stack4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.ELU()
        )
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.embedding(x).float()
        x = torch.cat((torch.sum(x,1) ,torch.mean(x,1)) , axis =1)
        x = self.linear_relu_stack1(x)
        x = self.linear_relu_stack2(x)
        x = self.linear_relu_stack3(x)
        x =  x + self.linear_relu_stack4(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

## Model1 - Modified Model0 architecture to use RNN
class Model1(nn.Module):
  def __init__(self,pretrained_embedding, embed_size, hidden_size, word2int):
    super(Model1,self).__init__()
    self.hidden_size = hidden_size
    self.embed_size = embed_size
    self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=word2int['<PAD>'])
  
    self.GRU = nn.GRU(embed_size,hidden_size ,batch_first = True, bidirectional = True,num_layers = 4 , dropout = 0.3)
    self.conv1D = nn.Sequential(
      nn.Conv1d(1,1,kernel_size=3,padding='same'),
      nn.MaxPool1d(3,stride = 2,padding = 1)
    )
    self.linear_relu1 = nn.Sequential(
      nn.Linear(hidden_size,hidden_size*2),
      nn.LayerNorm(hidden_size*2),
      nn.Dropout(0.2),
      nn.ReLU()
    ) 
    self.linear_relu2 = nn.Sequential(
      nn.Linear(hidden_size*2,hidden_size),
      nn.LayerNorm(hidden_size),
      nn.Dropout(0.2),
      nn.ReLU()
    ) 
    self.linear_sigmoid = nn.Sequential(
      nn.Linear(hidden_size,1),
      nn.Sigmoid()
    )

  def forward(self,x):
    x = self.embedding(x)
    x = nn.Dropout(0.2)(x)
    output,_  = self.GRU(x.float())
    output = nn.Dropout(0.2)(output)
    output_mean = torch.mean(output,1).unsqueeze(1)
    x = self.conv1D(output_mean)
    x = self.linear_relu1(x)
    x = self.linear_relu2(x)
    logits = self.linear_sigmoid(x)
    return logits

## Model2 - use packed padded sequence to avoid giving padded word vectors as output.
class Model2(nn.Module):
  def __init__(self,pretrained_embedding , embed_size, hidden_size , word2int):
    super(Model2,self).__init__()
    self.hidden_size = hidden_size
    self.embed_size = embed_size
    self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False , padding_idx=word2int['<PAD>'])
    self.GRU = nn.LSTM(embed_size,hidden_size ,batch_first= True ,bidirectional = True ,num_layers = 3 , dropout = 0.3)
    self.conv1D = nn.Sequential(
      nn.Conv1d(1,1,kernel_size=9,padding='same'),
      nn.MaxPool1d(9,stride = 2,padding = 4)
    )
    self.linear_relu1 = nn.Sequential(
      nn.Linear(hidden_size,hidden_size*2),
      nn.LayerNorm(hidden_size * 2),
      nn.Dropout(0.3),
      nn.ReLU()
    ) 
    self.linear_relu2 = nn.Sequential(
      nn.Linear(hidden_size*2,hidden_size),
      nn.LayerNorm(hidden_size),
      nn.Dropout(0.3),
      nn.ReLU()
    ) 
    self.linear_sigmoid = nn.Sequential(
      nn.Linear(hidden_size,1),
      nn.Sigmoid()
    )

  def forward(self,x,x_lengths):
    x = self.embedding(x)
    x = nn.Dropout(0.4)(x)
    x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True,enforce_sorted=False)
    output,_  = self.GRU(x.float())
    x, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
    output_mean = torch.mean(x,1).unsqueeze(1)
    x = self.linear_relu1(self.conv1D(output_mean))
    x = self.linear_relu2(x)
    logits = self.linear_sigmoid(x)
    return logits
