from collections import Counter
import numpy as np
import contractions
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))


# text cleaning

class CleanText:
    def __init__(self):
        pass

    def rm_link(self,text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    # def rm_punct2(self,text):
    #     return re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
    
    def rm_html(self,text):
        return re.sub(r'<[^>]+>', '', text)

    def space_bt_punct(self,text):
        pattern = r'([.,!?-])'
        s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
        s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
        return s
        
    def rm_punct2(self,text):
        return re.sub(r'[\"\#\$\%\&\-\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~\.\,\!\?]', ' ', text)
    
    # def rm_html(self,text):
    #     return re.sub(r'<[^>]+>', '', text)

    # def space_bt_punct(self,text):
    #     # pattern = r'([.,!?-])'
    #     # pattern = r'([.,!?])'
    #     # s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
    #     # s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
    #     # return s
    #     return text

    def rm_number(self,text):
        return re.sub(r'\d+', '', text)

    def rm_whitespaces(self,text):
        return re.sub(r' +', ' ', text)

    def rm_nonascii(self,text):
        return re.sub(r'[^\x00-\x7f]', r'', text)

    def rm_emoji(self,text):
        emojis = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE
        )
        return emojis.sub(r'', text)

    def spell_correction(self,text):
        return re.sub(r'(.)\1+', r'\1\1', text)

    def contraction(self,text):
        expanded_words = []   
        for word in text.split():
            # using contractions.fix to expand the shortened words
            expanded_words.append(contractions.fix(word))  
        return ' '.join(expanded_words)

    def clean_pipeline(self,text):
        contract_text = self.contraction(text)    
        no_link = self.rm_link(contract_text)
        no_html = self.rm_html(no_link)
        space_punct = self.space_bt_punct(no_html)
        no_punct = self.rm_punct2(space_punct)
        no_number = self.rm_number(no_punct)
        no_whitespaces = self.rm_whitespaces(no_number)
        no_nonasci = self.rm_nonascii(no_whitespaces)
        no_emoji = self.rm_emoji(no_nonasci)
        spell_corrected = self.spell_correction(no_emoji)
        return spell_corrected


# Text preprocessing
class PreprocessText:
    def tokenize(self,text):
        return word_tokenize(text)

    def removePOSTAGS(self,tokens):
        tagged = dict(nltk.pos_tag(tokens))
        remove  = ['DT' , 'IN' , 'PRP' ,'PRP$']
        words = []
        for w , t in tagged.items():
            if t not in remove:
                words.append(w)
        return words


    def rm_stopwords(self,text):
        return [i for i in text if i not in stopwords]

    def lemmatize(self,text):
        lemmatizer = WordNetLemmatizer()    
        lemmas = [lemmatizer.lemmatize(t) for t in text]
        # make sure lemmas does not contains sotpwords
        return self.rm_stopwords(lemmas)

    def preprocess_pipeline(self,text):
        tokens = self.tokenize(text)
        
        # no_stopwords = self.rm_stopwords(tokens)
        # lemmas = self.lemmatize(no_stopwords)
        lemmas = tokens
        return ' '.join(lemmas).lower()

# Create Vocabulary
class Vocab:
    def __init__(self, text):
        self.text = text
        self.getVocab()
        self.getIntToWord()
        self.getWordToInt()

    def getVocab(self):
        vocab = self.text.split()
        counter = Counter(vocab)
        self.vocab = sorted(counter, key=counter.get, reverse=True)

    def getIntToWord(self):
        self.int2word = { 0 :'<PAD>'}
        self.int2word.update(dict(enumerate(self.vocab,1)))

    def getWordToInt(self):
        self.wordToint = {word:index for index,word in self.int2word.items()}


# Create Training Set from Tiny Corpus

class CreateTrainSet:
    def __init__(self,corpus ,word2int, k = 20):
        '''
            -- arguments --
            corpus  : the ordered vector of the words present in original text
            word2int: vocabulary 
            k       : Window size of creating context

        '''
        self.corpus = corpus
        self.word2int = word2int
        self.k = k
        self.vocabsize = len(word2int)
        self.createInputOutput()

    def __createMultiHotVector(self,context):
        k = len(context)
        tempout = np.zeros(self.vocabsize)

        for i in range(k):
            if i != k//2 and context[i] in self.word2int.keys():
                tempout[self.word2int[context[i]]] = 1
        
        return tempout


    def createInputOutput(self):
        i= 0 
        input = []
        label = []
        while True:
            context = self.corpus[i : i+self.k]
            midword = context[self.k//2]
            if midword in self.word2int.keys():
                input.append(self.word2int[midword])
                label.append(self.__createMultiHotVector(context))

            if((i+self.k) == len(self.corpus)):
                break
            i+=1

        self.input, self.label =  np.array(input) , np.array(label)


def pad_features(reviews, pad_id, seq_length=128):
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features

# import jellyfish
# from fuzzywuzzy import fuzz
# def compare_keywords(word, keywords ,distance = 'jw'):
#     m = 0
#     keyword = None
#     for key in keywords:
#         if distance == 'jw' :
#             sim = jellyfish.jaro_winkler_similarity(word, key)
#             if sim > m:
#                 m = sim
#                 keyword = key
#         elif distance == 'hm':
#             sim = 9 - jellyfish.hamming_distance(word, key)
#             if sim > m:
#                 m = sim
#                 keyword = key
#         elif distance == 'tsr':
#             sim = fuzz.token_set_ratio(word,key)
#             if sim > m:
#                 m = sim
#                 keyword = key
#         else:
#             return 'Incorrect Distance',-1
            
#     return keyword,m


