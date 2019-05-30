import gensim
import os
import codecs
import ic
import logging
import pandas as pd 
import numpy as np


# Log output. Also useful to show program is doing things
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# models trained using gensim implementation of word2vec
print 'Loading models...'
model_source = gensim.models.Word2Vec.load('model_jp_wiki')
model_target = gensim.models.Word2Vec.load('model_en_wiki')


print 'Reading training pairs...'
word_pairs = codecs.open('ja-en-word-pairs.csv', 'r', 'utf-8')

pairs = pd.read_csv(word_pairs)

print 'Removing missing vocabulary...'

missing = 0

for n in range (len(pairs)):
	if pairs['source'][n] not in model_source.vocab or pairs['target'][n] not in model_target.vocab:
		missing = missing + 1
		pairs = pairs.drop(n)

pairs = pairs.reset_index(drop = True)
print 'Amount of missing vocab: ', missing

# make list of pair words, excluding the missing vocabs 
# removed in previous step
pairs['vector_source'] = [model_source[pairs['source'][n]] for n in range (len(pairs))]
pairs['vector_target'] = [model_target[pairs['target'][n]] for n in range (len(pairs))]

# first 5000 from both languages, to train translation matrix
source_training_set = pairs['vector_source'][:5000]
target_training_set = pairs['vector_target'][:5000]

matrix_train_source = pd.DataFrame(source_training_set.tolist()).values
matrix_train_target = pd.DataFrame(target_training_set.tolist()).values

print 'Generating translation matrix'

# Matrix W is given in  http://stackoverflow.com/questions/27980159/fit-a-linear-transformation-in-python
translation_matrix = np.linalg.pinv(matrix_train_source).dot(matrix_train_target).T
print 'Generated translation matrix'

# Returns list of topn closest vectors to vectenter
def most_similar_vector(self, vectenter, topn=5):
    self.init_sims()
    dists = np.dot(self.syn0norm, vectenter)
    if not topn:
        return dists
    best = np.argsort(dists)[::-1][:topn ]
        # ignore (don't return) words from the input
    result = [(self.index2word[sim], float(dists[sim])) for sim in best]
    return result[:topn]

def top_translations(w,numb=5):
    val = most_similar_vector(model_target,translation_matrix.dot(model_source[w]),numb)
    #print 'traducwithscofres ', val
    return val


def top_translations_list(w, numb=5):
    val = [top_translations(w,numb)[k][0] for k in range(numb)]
    return val

temp = 1
#top_matches = [ pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(5000,5003)] 

# print out source word and translation
def display_translations():
    for word_num in range(range_start, range_end):
        source_word =  pairs['source'][word_num]
        translations = top_translations_list(pairs['source'][word_num]) 
        print source_word, translations

# range to use to check accuracy
range_start = 5000
range_end = 6000

#display_translations()

# now we can check for accuracy on words 5000-6000, 1-5000 used to traning
# translation matrix

# returns matrix of true or false, true if translation is accuracy, false if not
# accurate means the first translation (most similiar vector in target language)
# is identical
accuracy_at_five = [pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(range_start, range_end)]
print 'Accuracy @5 is ', sum(accuracy_at_five), '/', len(accuracy_at_five)

accuracy_at_one = [pairs['target'][n] in top_translations_list(pairs['source'][n],1) for n in range(range_start, range_end)]
print 'Accuracy @1 is ', sum(accuracy_at_one), '/', len(accuracy_at_one)


