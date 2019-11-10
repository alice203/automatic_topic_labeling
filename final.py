import glob
import os
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import numpy as np
import pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise
import pyLDAvis.gensim as gensimvis
import pyLDAvis

path = "/Users/aliciahorsch/Anaconda/Master Thesis/"
wd = os.chdir(path)
wd = os.getcwd()

# Ted Talk scripts
main = pd.read_csv('ted_main.csv')
transcripts = pd.read_csv('transcripts.csv')

data = pd.merge(main, transcripts, how='inner', on='url')
data.count()
#View first lines of transcripts
data["transcript"].head()

#Subset data to X (transcripts) and y (manually annotated topic tags)
X = data.as_matrix(columns=data.columns[-1:])
y = data.as_matrix(columns=data.columns[-5:-4])

# Set aside a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.185, random_state=42)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#Clean and unify manually annotated topic tags
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

y = clean_y(y_train)

#EDA of main corpus
def min_tags(y):
    mi = 100
    i = 0 
    for index, item in enumerate(y):
        if len(item) < mi:
            mi = len(item)
            i = index
        else:
            continue
    return i, mi

def max_tags(y):
    ma = 0
    i = 0
    for index,item in enumerate(y):
        if len(item) > ma:
            ma = len(item)
            i = index
        else:
            continue
    return i, ma

#How many tags per transcript?
index_mi, mi = min_tags(y)
index_ma, ma = max_tags(y)
print(mi)
print(ma)

def create_frequency_table(tags):
    dic = {}    
    #Unpack list
    new_list = []
    for l in tags:
        for tag in l:
            if tag in new_list:
                continue
            else:
                new_list.append(tag)    
    #Create dictionary
    index = 0
    for item in new_list:
        dic[item] = index
        index += 1        
    #Create frequency table
    table = np.zeros((len(dic)))
    for l in tags:
        for tag in l:
            table[dic[tag]] += 1   
    nl = sorted(new_list)   
    return nl, dic, table

#Frequency table of tags
unique_tags, dic, ft = create_frequency_table(y)
print(ft)

#Number of total tags
tt = int(ft.sum())
print(tt)
#Average occurence of topic
ao = ft.mean()
#print(ao)

#Create reverse dictionary
def reverse_dic(dictionary):
    dic_new = {}
    for key, value in dictionary.items():
        dic_new[value] = key
        
    return dic_new

rev_dic = reverse_dic(dic)

def most_used_tags(frequency_table, reverse_dictionary, number):
    l = []
    for item in range(number):
        index = frequency_table.argmax()
        print(index)
        l.append(index)
        frequency_table[index] = 0
    #Decode
    decode = []
    for i in l:
        d = reverse_dictionary[i]
        decode.append(d)
    return decode

three_most_used = most_used_tags(ft, rev_dic,3)
print(three_most_used)


# Create a subset of observations with more balanced classes
def create_subset(X, y, frequency_table, dictionary, thresh, limit):
    keep = []
    for index in range(len(X)):
        tmp = y[index]
        count = []
        for element in tmp:
            decode = dictionary[element]
            #print(frequency_table[decode])
            if frequency_table[decode] < thresh:
                count.append(1)
            else:
                count.append(0)
            
        if sum(count) > 0:
            keep.append(index)
        else:
            continue            
    return keep

k = create_subset(X_train,y,ft,dic,20,100)

X_new = X_train[k]
#print(len(X_new))
y_new = y_train[k]
#print(len(y_new))

y_2 = clean_y(y_new)
#print(y_2)
nl, _, ft2 = create_frequency_table(y_2)

#Clean X: transcripts
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

X = clean_X(X_new)

#Delete stop words and lemmatize
stop_words = set(stopwords.words('english'))
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

X = process_words(X, stop_words)

########## ----------------------------------------------- LDA Model
#Source from https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

# Create a dictionary representation of the documents.
dictionary = Dictionary(X)

## Remove rare and common tokens. Filter out words that occur less than 10 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in X]

#Visualize corpus/BOW-representation
C = corpus2dense(corpus, len(dictionary))
s = C.sum(axis=0)
#print(C)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 20
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
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

topic_prob = model.print_topics()
#model.show_topics(num_topics=10, num_words=100, log=False, formatted=True)

#Top terms for topic 0
topic_0 = model.get_topic_terms(0, topn=10)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from gensim.models.coherencemodel import CoherenceModel

#### Topic-term matrix
top_term = model.get_topics()
#print(top_term.shape)

#Generate word list based on word probability
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

gen_word_list = generate_wordlist(top_term,dictionary)

#### Document-topic matrix
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

doc_top = get_doc_top_matrix(corpus, 0.0).reshape(2010,20)

print('Topic-Term shape is ', matrix.shape)
print('Doc-Topic shape is ', doc_top.shape)

#### Visualization
#ATTENTION VISUALIZATION is not identical with the topic order above. E.g. My topic 1 is "bacteria" containing "air" as most prominent word
#This is topic 7 below
#Source from https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
vis_data = gensimvis.prepare(model, corpus, dictionary)
pyLDAvis.display(vis_data)

########## ----------------------------------------------- Load reference text
#Load Wikipedia reference texts
def load_scripts(path):
    strings = []
    for seq_path in sorted(glob.glob(path)):
        #print(seq_path)
        proxy = open(seq_path).read()
        proxy = proxy.replace('\n', " ")
        strings.append(proxy)
    return strings

ref = load_scripts(wd+"/wiki/*.txt")

#Load Wikipedia labels
def extract_topics(filepath):
    filenames = []
    import glob, os
    os.chdir(filepath)
    for file in sorted(glob.glob("*.txt")):
        file = file.replace(".txt","")
        filenames.append(file)
    return filenames

wiki_labels = extract_topics("/Users/aliciahorsch/Anaconda/Master Thesis/wiki")

# Clean Wikipedia scripts
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

#Remove stopwords and lemmatize
ref2 = clean_wiki(ref)
ref2 = process_words(ref2, stop_words)

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

topics_fin = label_topics(gen_word_list, ref2, 2, wiki_labels)

########## ----------------------------------------------- Evaluation 
#(Assign documents with topics and compare with manual tags)
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

end = flag_documents(doc_top, 2, topics_fin)

def evaluation(y_lda, y):
    counter = 0
    for item in range(len(y_lda)):
        if y_lda[item][1] in y[item]:
            counter += 1
        else:
            continue
    return (counter/len(y_lda))

accuracy = evaluation(end, y_2)
print(accuracy)

