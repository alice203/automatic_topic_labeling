#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import pyLDAvis.gensim as gensimvis
import pyLDAvis


# In[2]:


wd = os.getcwd()
print(wd)


# Data from Kaggle

# In[3]:


main = pd.read_csv('ted_main.csv')
print(main.shape)


# In[4]:


transcripts = pd.read_csv('transcripts.csv')


# In[5]:


print(transcripts.shape)


# In[7]:


data = pd.merge(main, transcripts, how='inner', on='url')


# In[8]:


data.count()


# In[9]:


print(data.shape)


# In[15]:


data["transcript"].head()


# In[10]:


X = data.as_matrix(columns=data.columns[-1:])


# In[11]:


print(X.shape)


# In[12]:


y = data.as_matrix(columns=data.columns[-5:-4])


# In[13]:


print(y.shape)
print(y[:2])


# Set aside a test set

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.185, random_state=42)


# In[15]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


print(y_train[:2])


# In[19]:


def clean_y(y):
    y_new = []
    for array in y:
        for string in array:
            clean = string.replace("[", "")
            clean = clean.replace("]", "")
            clean = clean.replace("'", "")
            clean = clean.replace(",", "")
            clean = clean.lower()
            clean = clean.split() #list of words
            y_new.append(clean)

    y_new = np.array(y_new)
    return y_new


# In[20]:


y = clean_y(y_train)


# In[21]:


print(y.shape)


# Comparison of X and y (Fits text to label?)

# In[33]:


print(y[2])


# In[34]:


print(X[2])


# Clean transcripts

# In[23]:


print(X)


# In[24]:


def clean_X(X):
    new_X = []
    for array in X:
        for string in array:
            clean = string.lower()
            clean = clean.replace("(laughter)", "")
            clean = clean.replace("(music)", "")
            clean = clean.replace("(applause)", "")
            clean = clean.replace("(cheering)", "")
            clean = clean.replace(",", " ")
            clean = clean.replace(".", " ")
            clean = clean.replace("!", " ")
            clean = clean.replace("?", " ")
            clean = clean.replace(";", " ")
            clean = clean.replace(":", " ")
            clean = clean.replace('"', " ")
            clean = clean.replace('—', " ")
            clean = clean.replace('-', " ")
            clean = clean.replace('(', " ")
            clean = clean.replace(')', " ")

            #To be
            clean = clean.replace("i'm", "i am")
            clean = clean.replace("you're", "you are")
            clean = clean.replace("he's", "he is")
            clean = clean.replace("she's", "she is")
            clean = clean.replace("it's", "it is")
            clean = clean.replace("we're", "we are")
            clean = clean.replace("they're", "they are")
            clean = clean.replace("it's", "it is")

            #Auxilary words
            clean = clean.replace("i'd", "i would")
            clean = clean.replace("you'd", "you would")
            clean = clean.replace("he'd", "he would")
            final = clean.replace("she'd", "she would")
            final = final.replace("we'd", "we would")
            final = final.replace("they'd", "they would")

            #To have
            final = final.replace("i've", "i have")
            final = final.replace("you've", "you have")
            final = final.replace("we've", "we have")
            final = final.replace("they've", "they have")

            #Future
            final = final.replace("i'll", "i will")
            final = final.replace("you'll", "you will")
            final = final.replace("we'll", "we will")
            final = final.replace("they'll", "they will")
            final = final.replace("it'll", "it will")
            final = final.replace("he'll", "he will")
            final = final.replace("she'll", "she will")

            #Negative
            final = final.replace("didn't", "did not")
            final = final.replace("don't", "do not")
            final = final.replace("doesn't", "does not")
            final = final.replace("wasn't", "was not")
            final = final.replace("weren't", "were not")
            final = final.replace("won't", "will not")
            final = final.replace("wouldn't", "would not")
            final = final.replace("shouldn't", "should not")
            final = final.replace("couldn't", "could not")
            final = final.replace("haven't", "have not")
            final = final.replace("hasn't", "has not")

            #Questions
            final = final.replace("what's", "what is")
            final = final.replace("where's", "where is")
            final = final.replace("who's", "who is")
            final = final.replace("how's", "how is")
            final = final.replace("why's", "why is")

            #Others
            final = final.replace("that's", "that is")
            final = final.replace("there's", "there is")
            final = final.replace("here's", "here is")
            final = final.replace("'", " ")
            final = final.split()
            
            new_X.append(final)
            
    return new_X      


# In[25]:


X = clean_X(X_train)


# In[26]:


print(len(X))


# In[27]:


print(X[:1])


# Delete stop words and lemmatize

# In[28]:


stop_words = set(stopwords.words('english'))


# In[29]:


def process_words(word_list, stop_words):
    WNL = WordNetLemmatizer()
    porter = PorterStemmer()
    a = []
    for script in word_list:
        tmp = []
        for word in script:
            if word not in stop_words:
                tmp.append(WNL.lemmatize(word))
        a.append(tmp)
    return a


# In[35]:


X = process_words(X, stop_words)


# In[36]:


#print(X[:1])


# In[37]:


print(len(X))


# LDA Model

# In[39]:


#Source from https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(X)

# Filter out words that occur less than 10 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)


# In[177]:


print(dictionary[4])
print(len(dictionary))


# In[178]:


print((X[0]))


# In[168]:


# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in X]


# In[171]:


from gensim.matutils import corpus2dense
a = corpus2dense(corpus, len(dictionary))
b = a.sum(axis=0)
print(b)


# In[174]:


print(corpus[0])


# In[113]:


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# In[196]:


# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 20
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
    minimum_probability=0.0)


# In[197]:


print(model.print_topics())


# In[198]:


model.show_topics(num_topics=10, num_words=100, log=False, formatted=True)


# In[199]:


#Top terms for topic 0
model.get_topic_terms(0, topn=10)


# In[200]:


print(dictionary[158])


# In[201]:


#Topic-term matrix
top_term = model.get_topics()
print(top_term[:1])


# In[202]:


print(top_term.shape)


# Generate word list based on word probability

# In[203]:


def generate_wordlist(term_topic_matrix, dictionary):
    a = []
    matrix = term_topic_matrix*1000
    n_top = len(term_topic_matrix)
    
    for item in range(n_top):
        tmp = matrix[item]
        l = []
        while len(l) <= 500 :
            index = tmp.argmax()
            num = int(round(tmp[index]))
            #print(num)
            for count in range(num):
                l.append((dictionary[index]))
            tmp[index]=0
        a.append(l)
    return a


# In[204]:


f = generate_wordlist(top_term,dictionary)
print(f[0])
#print(f[1])


# Load reference text

# In[69]:


#Load rtf_scripts
def load_scripts(path):
    strings = []
    for seq_path in sorted(glob.glob(path)):
        #print(seq_path)
        proxy = open(seq_path).read()
        proxy = proxy.replace('\n', " ")
        strings.append(proxy)
    return strings


# In[70]:


ref = load_scripts(wd+"/wiki/*.txt")


# In[71]:


print(ref[15:16])


# In[72]:


def extract_topics(filepath):
    filenames = []
    import glob, os
    os.chdir(filepath)
    for file in sorted(glob.glob("*.txt")):
        file = file.replace(".txt","")
        filenames.append(file)
    return filenames


# In[73]:


topics = extract_topics("/Users/aliciahorsch/Anaconda/Master Thesis/wiki")
print(topics)


# Clean string

# In[74]:


#Source: https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python/14598135

def remove_sources(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret


# In[82]:


def clean_wiki(list_of_strings):
    new = []
    for string in list_of_strings:
        clean = string.lower()
        clean = clean.replace(",", " ")
        clean = clean.replace(".", " ")
        clean = clean.replace("!", " ")
        clean = clean.replace("?", " ")
        clean = clean.replace(";", " ")
        clean = clean.replace(":", " ")
        clean = clean.replace('"', " ")
        clean = clean.replace("'", " ")
        clean = clean.replace('–', " ")
        clean = clean.replace('-', " ")
        clean = clean.replace('(', " ")
        clean = clean.replace(')', " ")
        
        #Remove wiki-words
        clean = clean.replace("main article", "")
        clean = clean.replace("[citation needed]", "")
        clean = clean.replace("see also", "")
        clean = clean.replace("[edit]", "")


        #Delete Sources
        clean = remove_sources(clean)
        clean = clean.split()
        
        new.append(clean)
    return new


# In[83]:


#Remove stopwords and lemmatize
ref2 = clean_wiki(ref)
ref2 = process_words(ref2, stop_words)


# In[84]:


print(ref2[67:68])


# In[85]:


#Parameters
print(len(ref2))
print(len(f))


# Find similarity

# In[205]:


import sklearn.metrics.pairwise


# In[206]:


#Source: Partly from Data Processing Advanced Notebooks (D.Hendrickson)

def label_topics(generated_word_list, reference_word_list, num_topics, ref_topics):
    #print(len(reference_word_list))
    #print(len(generated_word_list))
    c = []
    tfidf_vectorizer = TfidfVectorizer(min_df=0.1, max_df= 0.88)
    
    for item in range(len(generated_word_list)):
        proxy = reference_word_list[:]
        #print(len(proxy))
        proxy.append(generated_word_list[item])
        #print(len(proxy))
        
        #For TF-IDF, need a list of sentences instead of list of words
        tmp = []
        for example in proxy:
            sentence = " ".join(example)
            tmp.append(sentence)
        
        term_freq_matrix = tfidf_vectorizer.fit_transform(tmp)
        M = term_freq_matrix.toarray()
        
        #Cosine similarity between topic and reference text
        cos = sklearn.metrics.pairwise.cosine_similarity(M)
        #print(cos.shape)
        
        index = cos.shape[0]
        #print(index)
        
        #Assign topic with label based on cosine-matrix
        num_most_similar_scripts = num_topics
        
        cos[index-1,index-1] = 0

        # work with this row of the similarity matrix
        tmp_2 = cos[index-1,]
        #print(tmp.shape)

        l = []
        l.append(item)
        # find most similar scripts
        for i in range(num_most_similar_scripts):
            #find max index
            index_2 = tmp_2.argmax()
            inner_list = ref_topics[index_2]
            l.append(inner_list)
            
            # set this similarity to 0 so it isn't found again
            tmp_2[index_2] = 0
        
        print(l)
        la = np.array(l)
        c.append(la)  
    return np.array(c)


# In[207]:


p = label_topics(f, ref2, 2, topics)
print(p.shape)


# In[208]:


print(y[1])


# In[209]:


print(X[1])


# Doc-topic matrix

# In[210]:


doc_top = model.get_document_topics(corpus[1], minimum_probability=0.0)
print(doc_top)


# In[211]:


print(len(X[0]))


# In[129]:


print(corpus[0])


# In[130]:


print(dictionary[3])


# In[212]:


def get_doc_top_matrix(corpus, min_prob=0.0):
    #print(len(corpus))
    proxy = []
    for i in range(len(corpus)):
        tmp = model.get_document_topics(corpus[i], minimum_probability=min_prob)
        #print(tmp)
        tmp_2 = []
        for item in tmp:
            tmp_2.append(item[1:])
        proxy.append(tmp_2)
    
    return np.array(proxy)
        


# In[214]:


doc_top = get_doc_top_matrix(corpus, 0.0).reshape(2010,20)


# In[215]:


print(doc_top[:1])


# In[134]:


print('Topic-Term shape is ', matrix.shape)
print('Doc-Topic shape is ', doc_top.shape)


# Visualization

# In[ ]:


#ATTENTION VISUALIZATION is not identical with the topic order above. E.g. My topic 1 is "bacteria" containing "air" as most prominent word
#This is topic 7 below


# In[94]:


#Source from https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb

vis_data = gensimvis.prepare(model, corpus, dictionary)
pyLDAvis.display(vis_data)


# Evaluation (Assign documents with topics and compare with manual tags)

# In[216]:


def flag_documents(doc_top_matrix, num_flags, topic_flag_matrix):
    n = []
    for i in range(len(doc_top_matrix)):
        m = []
        m.append(i)
        for j in range(num_flags):
            tmp = doc_top_matrix[i,]
            index = tmp.argmax()
            #print(topic_flag_matrix[index][1])
            m.append(topic_flag_matrix[index][1])
            doc_top_matrix[i,index]= 0
        n.append(np.array(m))
    return np.array(n)


# In[217]:


end = flag_documents(doc_top, 2, p)
print(end.shape)


# In[218]:


print(end)


# In[221]:


print(y)


# In[222]:


def evaluation(y_lda, y):
    counter = 0
    for item in range(len(y_lda)):
        if y_lda[item][1] in y[item]:
            counter += 1
        else:
            continue
    return (counter/len(y_lda))


# In[223]:


accuracy = evaluation(end, y)
print(accuracy)

