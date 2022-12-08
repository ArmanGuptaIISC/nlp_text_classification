## Model Experimentation For Text Classification
Purpose Of Experiment - To understand  how RNN Architecture helps in classification task of temporal data like Text. 

Experiment done in Month - August

Models-
1. Model0 -  Basic DAN kind of structure
2. Model1 -  Modified Model0 architecture to use RNN
3. Model2 -  Use packed padded sequence to avoid giving padded word vectors as output.

Dataset-
1. Data0 -\
    i. Train.neg - Contain Negative sentiments \
    ii. Train.pos - Contain Positive Sentiments \
    iii. TestData - Contain Test File having first half positive and second half negative sentiments
2. Data1 -\
    Imdb Dataset divided into training and testing dataset.

Commands-

usage:\
python main.py \
[--batchsize BATCHSIZE] \
[--seqlen SEQLEN] \
[--glovepath GLOVEPATH]\
[--dnum DNUM]\
[--cuda_num CUDA_NUM]\
[--epochs EPOCHS]\
[--model MODEL]\
[--patience PATIENCE]

<pre>
arguments:
-h, --help          show this help message and exit
--batchsize         Batch Size
--seqlen            Seq len of each word vector after padding and truncating
--glovepath         Path to Glove Embedding file
--dnum              Dataset are numbered. This take which dataset I want to use
--cuda_num          device number
--epochs            Number of Epochs
--model             Models are numbered. Select the appropriate model number.
--patience          How much you want to wait before early stopping

</pre>