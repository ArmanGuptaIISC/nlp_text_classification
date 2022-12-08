from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from util import CleanText , PreprocessText , pad_features
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch 
from torch import nn
from torch.utils.data import DataLoader , TensorDataset
from sklearn.model_selection import train_test_split
from models import *
from training_val_fn import train_epoch, val_epoch
import argparse

parser = argparse.ArgumentParser(description= 'Input the file path and model hyperparameters')
parser.add_argument('--batchsize' ,type = int, default= 256 ,help = 'Batch Size')
parser.add_argument('--seqlen' ,type = int ,default= 100, help = 'Seq len of each word vector after padding and truncating')
parser.add_argument('--glovepath', help = 'Path to Glove Embedding file')
parser.add_argument('--dnum', type =int , default = 0, help = 'Dataset are numbered. This take which dataset I want to use')
parser.add_argument('--cuda_num' , type = int , default= 0, help= 'Cuda device number')
parser.add_argument('--epochs' ,type= int , default= 50, help = 'Number of Epochs')
parser.add_argument('--model', type = int , default= 0 , help = "Models are numbered. Select the appropriate model number.")
parser.add_argument('--patience' , type = int, default= 6 , help =' How much you want to wait before early stopping')
args = parser.parse_args()
BATCH_SIZE = args.batchsize 
seq_length = args.seqlen
cuda_device_num = args.cuda_num
glove_file = args.glovepath
dataset_number = args.dnum
model_number = args.model

device = torch.device('cuda:'+str(cuda_device_num) if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

if dataset_number == 0:
    pos_file = './data/data0/Train.pos'
    neg_file = './data/data0/Train.neg'
    test_file = '/data/data0/TestData'
    ## Loading Training and Testing files 
    train_reviews = []
    label = []
    with codecs.open(pos_file,'r',encoding='utf-8',errors= 'ignore') as file:
        for line in file.readlines():
            train_reviews.append(line)
            label.append(1)

    with codecs.open(neg_file,'r',encoding='utf-8',errors= 'ignore') as file:
        for line in file.readlines():
            train_reviews.append(line)
            label.append(0)

    test_reviews = []
    with codecs.open(test_file , 'r',encoding='utf-8',errors= 'ignore') as file:
        for line in file.readlines():
            test_reviews.append(line)

    test_label = np.concatenate([np.ones(331),np.zeros(331)])

elif dataset_number == 1:
    data = pd.read_csv('./data/data1/train.csv')
    train_reviews = data['review'].to_numpy()
    label = data['sentiment'].apply(lambda x : 1 if x=='positive' else 0).to_numpy()
    test = pd.read_csv('./data/data1/test.csv', names = ['review','sentiment'])
    test_reviews = test['review'].to_numpy()
    test_label = test['sentiment'].apply(lambda x : 1 if x=='positive' else 0).to_numpy()
    
# Loading Glove Embeddings
word2vec = {}
word2int = {}
with open(glove_file) as f:
    for i, line in enumerate(f.readlines()):
        words = line.split()
        EMB_SIZE  = len(words) - 1
        word2vec[words[0]] = np.array([float(word) for word in words[1:]])
        word2int[words[0]] = i



# Preprocessing the train and test file
cleantext = CleanText()
preprocess = PreprocessText()
train_reviews = [cleantext.clean_pipeline(text) for text in train_reviews]
train_reviews = [preprocess.preprocess_pipeline(text) for text in train_reviews]
train_reviews = [cleantext.clean_pipeline(text).strip() for text in train_reviews]

test_reviews = [cleantext.clean_pipeline(text) for text in test_reviews]
test_reviews = [preprocess.preprocess_pipeline(text) for text in test_reviews]
test_reviews = [cleantext.clean_pipeline(text).strip() for text in test_reviews]

## Integrating the missing words to vocabulary and their random vectors from the given Data
corpus = ' '.join(train_reviews)
words = corpus.split()
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True)
remain_train_words = set(vocab).difference(word2int.keys())
random_vector_initialisation = np.random.normal(0,3 , size = (len(remain_train_words) , EMB_SIZE))

start = max(list(word2int.values())) + 1
remain_train_word_w2i = dict(zip(remain_train_words,np.arange(start , start+len(remain_train_words))))
word2int.update(remain_train_word_w2i)
word2int['<PAD>'] = max(list(word2int.values())) + 1

embedding_vectors = np.array(list(word2vec.values()))
embeddings = np.vstack((embedding_vectors,random_vector_initialisation))
embeddings = np.vstack((embeddings , np.zeros((1,EMB_SIZE))))

# Encoding Words using word2int 
train_reviews_enc = [[word2int[word] for word in review.split()] for review in tqdm(train_reviews)]
test_reviews_enc = [[word2int[word] for word in review.split() if word in word2int.keys()] for review in tqdm(test_reviews)]

train_feat = pad_features(train_reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)
test_feat = pad_features(test_reviews_enc , pad_id= word2int['<PAD>'], seq_length=seq_length)

assert len(train_feat) == len(train_reviews_enc)
assert len(test_feat) == len(test_reviews_enc)
assert len(train_feat[0]) == seq_length
assert len(test_feat[0]) == seq_length

label = np.array(label)
test_label = np.array(test_label)

## spliting train dataset into train and val set
x_train, x_val , y_train, y_val = train_test_split(train_feat, label,test_size= 0.1)


fulldataset = TensorDataset(torch.from_numpy(train_feat) , torch.from_numpy(label))
trainset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
valset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
testset = TensorDataset(torch.from_numpy(test_feat), torch.from_numpy(test_label))

fulltrainloader = DataLoader(fulldataset , shuffle = True , batch_size= BATCH_SIZE)
trainloader = DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE)
valloader = DataLoader(valset,shuffle= False ,batch_size=BATCH_SIZE)
testloader = DataLoader(testset,shuffle= False,batch_size= len(testset))

torch.manual_seed(42)
embedding_size = EMB_SIZE
hidden_size = EMB_SIZE 
pretrained_embedding = torch.from_numpy(embeddings)
if model_number == 0:
    model = Model0(pretrained_embedding=pretrained_embedding, in_feat= embedding_size , hidden_size= hidden_size, word2int= word2int)
    lr = 0.0005
elif model_number == 1:
    model = Model1(pretrained_embedding=pretrained_embedding,embed_size= embedding_size , hidden_size= hidden_size ,word2int= word2int)
    lr = 0.0005
elif model_number == 2:
    model = Model2(pretrained_embedding=pretrained_embedding,embed_size= embedding_size , hidden_size= hidden_size ,word2int= word2int)
    lr = 0.0005
    
print('Model Architecture : \n',model)


criterion = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)
grad_clip = 5
epochs = args.epochs
print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': epochs
}
es_limit = args.patience


# train loop
model = model.to(device)

epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

# early stop trigger
es_trigger = 0
val_loss_min = torch.inf

for e in epochloop:

    model.train()
    train_loss, train_acc = train_epoch(model ,device , trainloader, criterion, optim, grad_clip, epochloop)

    history['train_loss'].append(train_loss / len(trainloader))

    model.eval()
    val_loss , val_acc = val_epoch(model, device ,valloader, criterion , epochloop)
    history['val_loss'].append(val_loss / len(valloader))
    history['val_acc'].append(val_acc / len(valloader))


    # add epoch meta info
    epochloop.set_postfix_str(f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

    # print epoch
    if (e+1) % print_every == 0:
        epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f}| Testing Loss: {val_loss / len(valloader):.3f} Test Acc: {val_acc / len(valloader):.3f}')
        epochloop.update()

    # save model if Training loss decrease
    if val_loss / len(valloader) <= val_loss_min:
        torch.save(model.state_dict(), './model.pt')
        val_loss_min = val_loss / len(valloader)
        es_trigger = 0
    else:
        epochloop.write(f'[WARNING] Testing loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
        es_trigger += 1

    # force early stop
    if es_trigger >= es_limit:
        epochloop.write(f'Early stopped at Epoch-{e+1}')
        # update epochs history
        history['epochs'] = e+1
        break


# Testing the model
test_loss , test_acc = val_epoch(model, device ,testloader,criterion , None)

print(f'Testing Loss : {test_loss/len(testloader)} and Testing Accuracy : {test_acc/len(testloader)}')