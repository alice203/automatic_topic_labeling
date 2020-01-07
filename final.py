'PLEASE ADJUST THE PATH (BELOW) TO YOUR INDIVIDUAL WORKING DIRECTORY'

import random
import decimal
import glob
import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from copy import deepcopy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy import stats

random.seed(0)

path = "/Users/aliciahorsch/Anaconda/Master Thesis/"
wd = os.chdir(path)
wd = os.getcwd()

### Pre-processing

######### ----------------------------------- Load data and clean data

# Ted Talk scripts
main = pd.read_csv('ted_main.csv')
transcripts = pd.read_csv('transcripts.csv')
data = pd.merge(main, transcripts, how='inner', on='url')

# Divide data to X (transcripts) and y (manually annotated topic tags)
X = data.as_matrix(columns=data.columns[-1:])
y = data.as_matrix(columns=data.columns[-5:-4])

# Clean transcripts
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
            clean = clean.replace('♫♫', " ")
            clean = clean.replace('’', " ")
            clean = clean.replace("'", " ")
            clean = clean.replace('´', " ")
            final = clean.split()            
            new_X.append(final)            
    return new_X 

#Delete stop words and lemmatize
stop_words = set(stopwords.words('english'))
def process_tokens(word_list, stop_words):
    WNL = WordNetLemmatizer()
    a = []
    for script in word_list:
        tmp = []
        for word in script:
            if word not in stop_words:
                word = WNL.lemmatize(word)
                if len(word) > 1:
                    tmp.append(WNL.lemmatize(word))
        a.append(tmp)
    return np.array(a)

# Clean and unify manually annotated topic tags
def clean_y(y):
    clean = []
    for liste in y:
        proxy = liste[0].split(",")
        tmp = []
        for item in proxy:
            new = item.lower()
            a = new.replace("[", "")
            a = a.replace("]", "")
            a = a.replace("'", "")
            a = a.replace('"', "")
            tmp.append(a.strip(" "))
        clean.append(tmp)
    return clean

# Unify labels
def unify_y(y):
    new = []
    for liste in y:
        tmp = []
        for item in liste:
            if item == "farming":
                tmp.append("agriculture")
                continue
            if item == "charter of compassion":
                tmp.append("compassion")
                continue
            if item == "funny":
                tmp.append("humor")
                continue
            if item == "illness":
                tmp.append("disease")
                continue
            if item == "vocals":
                tmp.append("singer")
                continue
            if item == "ted books":
                tmp.append("singer")
                continue
            if item == "ted books":
                continue
            if item == "ted brain trust":
                continue
            if item == "ted en español":
                continue
            if item == "ted fellows":
                continue
            if item == "ted prize":
                continue
            if item == "ted residency":
                continue
            if item == "ted-ed":
                continue
            if item == "tedmed":
                continue
            if item == "tednyc":
                continue
            if item == "tedx":
                continue
            if item == "tedyouth":
                continue 
            if item == "testing":
                continue 
            if item == "cyborg":
                continue
            if item == "testing":
                continue
            else:
                tmp.append(item)
        new.append(tmp)       
    new2 = []
    for ls in new:
        prx = []
        for element in ls:
            if element in prx:
                continue
            else:
                prx.append(element)
        new2.append(prx)    
    return np.array(new2)

X = clean_X(X)
X = process_tokens(X, stop_words)

y = clean_y(y)
y = unify_y(y)

# Reference corpus
# Load
def load_wiki(path):
    strings = []
    for seq_path in sorted(glob.glob(path)):
        proxy = open(seq_path).read()
        proxy = proxy.replace('\n', " ")
        strings.append(proxy)
    return strings

#Extract labels from file-name
def extract_topics(filepath):
    filenames = []
    import glob, os
    os.chdir(filepath)
    for file in sorted(glob.glob("*.txt")):
        file = file.replace(".txt","")
        filenames.append(file)
    return filenames

# Clean articles
# Source (1): https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python/14598135
def remove_parenth(test_str):
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
        #Numbers
        clean = clean.replace("0", "")
        clean = clean.replace("1", "")
        clean = clean.replace("2", "")
        clean = clean.replace("3", "")
        clean = clean.replace("4", " ")
        clean = clean.replace("5", " ")
        clean = clean.replace("6", " ")
        clean = clean.replace("7", " ")
        clean = clean.replace("8", " ")
        clean = clean.replace("9", " ")       
        #Remove wiki-words
        clean = clean.replace("main article", "")
        clean = clean.replace("[citation needed]", "")
        clean = clean.replace("see also", "")
        clean = clean.replace("[edit]", "")
        clean = clean.replace("v t e", "")
        #Delete Sources
        clean = remove_parenth(clean)
        clean = clean.split()     
        new.append(clean)
    return new

ref = load_wiki(wd+"/wiki/*.txt")
wiki_labels = extract_topics("/Users/aliciahorsch/Anaconda/Master Thesis/wiki")

# Clean articles, remove stopwords and lemmatize
ref = clean_wiki(ref)
ref = process_tokens(ref, stop_words)
ref = list(ref)

######### ---------------------------------------- EDA

# Main corpus
def distribution_labels(y):
    length = []
    for l in y:
        length.append(len(l))
    return np.array(length)

def frequency_table(tags):
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

label_distribution = distribution_labels(y)
unique_tags, dic, ft = frequency_table(y)

# Exploring labels
print("Label distribution")
print("Maximum number of labels per transcript: ", label_distribution.max())
print("Minimum number of labels per transcript: ", label_distribution.min())
print("The average amount of labels per transcript is ", int(np.quantile(label_distribution, 0.5)))
print("25% of transcripts have less or exactly ", int(np.quantile(label_distribution, 0.25)), " labels (25%-quantile)")
print("75% of transcripts have more or exactly ", int(np.quantile(label_distribution, 0.75)), " labels (75%-quantile)")
print()

# Exploring topics
print("Topic distribution")
print("Total amount of unique topics in corpus: ", len(ft))
print("Total amount of topics in corpus: ", int(ft.sum()))
print("Average amount of occurence of tag: ", int(ft.mean()))
print("25% of tags occur less than or exactly ", int(np.quantile(ft,0.25)), " times (25%-quantile)")
print("25% of tags occur more than or exactly ", int(np.quantile(ft,0.75)), " times (75% quantile)")

# Boxplot: Distribution of topic tags
csfont = {'fontname':'Times New Roman', 'fontsize':'14'}
plt.boxplot(ft)
plt.xlabel("Transcripts", **csfont)
plt.ylabel("Frequency of topics", **csfont)
#plt.show()
plt.savefig(wd+'/visuals/boxplot_averagetagamount.png', dpi = 500)

# Create reverse dictionary
def reverse_dic(dictionary):
    dic_new = {}
    for key, value in dictionary.items():
        dic_new[value] = key       
    return dic_new

def most_freq_tags(frequency_table, reverse_dictionary, number):
    amount = []
    l = []
    for item in range(number):
        index = frequency_table.argmax()
        l.append(index)
        amount.append(frequency_table[index])
        frequency_table[index] = 0
    #Decode
    decode = []
    for i in l:
        d = reverse_dictionary[i]
        decode.append(d)
    return amount, decode

def least_freq_tags(frequency_table, reverse_dictionary, number):
    amount = []
    l = []
    for item in range(number):
        index = frequency_table.argmin()
        l.append(index)
        amount.append(frequency_table[index])
        frequency_table[index] = 1000
    #Decode
    decode = []
    for i in l:
        d = reverse_dictionary[i]
        decode.append(d)
    return amount, decode

# 10 most frequent topics
ft1 = deepcopy(ft)
rev_dic = reverse_dic(dic)
frequency, most_used = most_freq_tags(ft1, rev_dic,10)
print(most_used)
print(frequency)

# 10 least frequent topics
ft2 = deepcopy(ft)
frequency_l, least_used = least_freq_tags(ft2, rev_dic,10)
print(least_used)
print(frequency_l)

# Barplot most frequent topics
height = frequency
bars = most_used
y_pos = np.arange(len(bars))
y_axes = range(0,750,100)
plt.bar(y_pos, height, color = ["midnightblue", "darkblue", "mediumblue", "royalblue", "cornflowerblue", 
                                "lightskyblue","skyblue", "powderblue", "lightsteelblue", "lavender"])
plt.xticks(y_pos, bars, rotation=90, **csfont)
plt.yticks(y_axes, **csfont)
plt.ylabel('Frequency', **csfont)
plt.xlabel('Topics', **csfont)
plt.tight_layout()
#plt.show()
plt.savefig(wd+'/visuals/Most_represented.png', dpi=500)

# Barplot least frequent topics
height = frequency_l
bars = least_used
y_pos = np.arange(len(bars))
y_axes = range(0,4,1)
plt.bar(y_pos, height, color = ["darkred", "maroon", "firebrick", "brown", "indianred", 
                                "lightcoral","rosybrown", "mistyrose", "salmon", "tomato"])
plt.xticks(y_pos, bars, rotation=90, **csfont)
plt.yticks(y_axes, **csfont)
plt.ylabel('Frequency', **csfont)
plt.xlabel('Topics', **csfont)
plt.tight_layout()
#plt.show()
plt.savefig(wd+'/visuals/Least_represented.png', dpi=500)

# Reference corpus
def average_length_reference_texts(reference_texts):
    length = []
    for item in reference_texts:
        length.append(len(item))
    l = np.array(length)
    return l.mean()

average_len_refs = average_length_reference_texts(ref)
print('Average length of reference texts: ', round(average_len_refs))


# ### Analysis (1.Part: LDA)

######### ---------------------------------------- LDA

# Source (2): https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

# Create a dictionary representation of the documents.
dictionary = Dictionary(X)
print("Number of unique tokens in corpus: ", len(dictionary))

# Filter out words that occur less than 10 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.5)
print("Number of unique tokens used for analysis: ", len(dictionary))

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in X]

model_list = []
temp = dictionary[0]
id2word = dictionary.id2token

model1 = LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=10, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model1)

model2 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=50, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model2)

model3 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=90, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model3)

model4 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=130, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model4)

model5 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=170, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model5)

model6 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=210, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model6)

model7 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=250, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model7)

model8 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=290, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model8)

model9 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=330, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model9)

model10 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=370, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model10)

model11 =LdaModel(corpus=corpus, id2word=dictionary.id2token, num_topics=410, chunksize=1000,
                alpha='auto', eta='auto', iterations=100, passes=10, eval_every=None, 
                minimum_probability=0.1, random_state=1)
model_list.append(model11)

######### ---------------------------------------- Topic-Term-Matrix

def list_of_topic_term_matrices(modellist):
    tt_matrices = []
    for model in modellist:
        topic_term = model.get_topics()
        tt_matrices.append(topic_term)
    return tt_matrices

tt_matrices = list_of_topic_term_matrices(model_list)

######### ---------------------------------------- Document-Topic-Matrix

def get_doc_top_matrix(model, corpus, min_prob=0.0):
    #print(corpus)
    proxy = []
    for i in range(len(corpus)):
        tmp = model.get_document_topics(corpus[i], minimum_probability=min_prob)
        #print(tmp)
        tmp_2 = []
        for item in tmp:
            tmp_2.append(item[1:])
        proxy.append(tmp_2)
    
    return np.array(proxy)

def doc_term_matrices(model_list, corpus):
    doc_tops = []
    for model in range(len(model_list)):
        doc_top = get_doc_top_matrix(model_list[model], corpus, 0.0)
        #print(doc_top.shape)
        doc_tops.append(doc_top.reshape(len(X),tt_matrices[model].shape[0]))
    return doc_tops

doc_tops = doc_term_matrices(model_list, corpus)

for index in range(len(doc_tops)):
    print('Model', index )
    print('Topic-Term shape is ', tt_matrices[index].shape)
    print('Doc-Topic shape is ', doc_tops[index].shape)
    print(' ')


# ### Analysis (2.Part: Prediciton of labels)

######### ---------------------------------------- Approach: Generating BOW

def generate_wordlist(term_topic_matrix, dictionary):
    a = []
    matrix = term_topic_matrix*10000
    n_top = len(term_topic_matrix)    
    for item in range(n_top):
        tmp = matrix[item]
        l = []
        while len(l) <= 3800 :
            index = tmp.argmax()
            num = int(round(tmp[index]))
            #print(num)
            if num == 0:
                break
            else: 
                for count in range(num):
                    l.append((dictionary[index]))
            tmp[index]=0
        a.append(l)
    return a

def gen_word_lists(top_term_matrices, dictionary):
    gen_word_lists = []
    for matrix in top_term_matrices:
        gen_word_list = generate_wordlist(matrix, dictionary)
        gen_word_lists.append(gen_word_list)
    return gen_word_lists

gwl = gen_word_lists(tt_matrices, dictionary)

#Source (3): Inspired from code used in course: Data Processing Advanced, Notebook 'Distance and Similarity',
#(D.Hendrickson), Tilburg University

def label_topics(generated_word_list, reference_word_list, num_topics, ref_topics):
    c = []
    tfidf_vectorizer = TfidfVectorizer()
    for item in range(len(generated_word_list)):
        proxy = reference_word_list[:]
        proxy.append(generated_word_list[item])
        #For TF-IDF, need a list of sentences instead of list of words
        tmp = []
        for example in proxy:
            sentence = " ".join(example)
            tmp.append(sentence)
        term_freq_matrix = tfidf_vectorizer.fit_transform(tmp)
        M = term_freq_matrix.toarray()
        #Cosine similarity between topic and reference text
        cos = sklearn.metrics.pairwise.cosine_similarity(M)
        index = cos.shape[0]
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
        la = np.array(l)
        c.append(la)  
    return np.array(c)

def topics_all(gen_word_lists):
    topics_all = []
    for model in gen_word_lists:
        p = label_topics(model, ref, 1, wiki_labels)
        topics_all.append(p)
    return topics_all

# Assign documents with topics
def flag_documents(doc_top_matrix, num_flags, topic_flag_matrix):
    n = []
    for i in range(len(doc_top_matrix)):
        m = []
        m.append(i)
        for j in range(num_flags):
            tmp = doc_top_matrix[i,]
            index = tmp.argmax()
            m.append(topic_flag_matrix[index][1])
            doc_top_matrix[i,index]= 0
        n.append(np.array(m))
    return np.array(n)

def final_tags(doc_tops, topics_all):
    ends = []
    for index in range(len(doc_tops)):
        end = flag_documents(doc_tops[index], 1, topics_all[index])
        ends.append(end)
    return ends

ta = topics_all(gwl)

final = final_tags(doc_tops, ta)

######### ---------------------------------------- Approach: Most representative documents

#For TF-IDF, need a list of sentences instead of list of words
def full_string(corpus):
    a = []
    for list_of_words in corpus:
        tmp = []
        string = " ".join(list_of_words)
        tmp.append(string)
        a.append(tmp)
    return a

def doc_list_per_topic(doc_top, documents, num=1):
    #Create list for topics
    topics = []
    final = []
    for count in range(doc_top.shape[1]):
        topics.append([])
        final.append("")    
    #Find documents representing the topic
    for doc in range(doc_top.shape[0]):
        for element in range(num):
            tmp = doc_top[doc,]
            index = tmp.argmax()
            topics[index].append(doc)
            tmp[index] = 0
    #Fill up gaps:
    for j, l in enumerate(topics):
        if len(l) == 0:
            prx = doc_top[:,j]
            k = prx.argmax()
            topics[j].append(k)   
    print(topics)
    #Prepare documents
    doc_collection = []
    list_of_strings = full_string(documents)
    for i,t in enumerate(topics):
        for item in range(len(t)):
            string =""
            ind = topics[i][item]
            string = list_of_strings[ind]
            final[i] += string[0]
    return final

def doc_list_per_topic(doc_top, documents, num=1):
    #Create list for topics
    topics = []
    final = []
    for topic in range(doc_top.shape[1]):
        prx = []
        for element in range(num):
            tmp = doc_top[:,topic]
            index = tmp.argmax()
            prx.append(index)
            tmp[index, ] = 0
        topics.append(prx)
    
    list_of_strings = full_string(documents)
    for i,t in enumerate(topics):
        string = ""
        for e in range(len(t)):
            j = topics[i][e]
            string += list_of_strings[j][0]
        final.append(string)
    return final

def topic_doc_list_allmodels(doc_tops, X, num_doc=1):
    final_text_per_topic = []
    for doc_top in doc_tops:
        topic_text = doc_list_per_topic(doc_top, X, num_doc)
        final_text_per_topic.append(topic_text)
    return final_text_per_topic

topic_texts = topic_doc_list_allmodels(doc_tops, X, 10)

# See source 3

def label_topics2(generated_word_list, reference_word_list, num_topics, ref_topics):
    c = []
    tfidf_vectorizer = TfidfVectorizer() 
    for item in range(len(generated_word_list)):
        proxy = []
        proxy = reference_word_list[:]
        proxy.append(generated_word_list[item])  
        term_freq_matrix = tfidf_vectorizer.fit_transform(proxy)
        M = term_freq_matrix.toarray()
        #Cosine similarity between topic and reference text
        cos = sklearn.metrics.pairwise.cosine_similarity(M)
        index = cos.shape[0]
        #Assign topic with label based on cosine-matrix
        num_most_similar_scripts = num_topics
        cos[index-1,index-1] = 0
        # work with this row of the similarity matrix
        tmp_2 = cos[index-1,]
        l = []
        l.append(item)
        # find most similar scripts
        for i in range(num_most_similar_scripts):
            #find max index
            index_2 = tmp_2.argmax()
            inner_list = ref_topics[index_2-1]
            l.append(inner_list)
            # set this similarity to 0 so it isn't found again
            tmp_2[index_2] = 0
        la = np.array(l)
        c.append(la)  
    return np.array(c)

#Data prep for TFIDF
def list_of_strings(ref):
    tmp = []
    for list_of_words in ref:
            string = " ".join(list_of_words)
            tmp.append(string)
    return tmp

def topics_all_2(string_lists):
    topics_all_2 = []
    for string_list in string_lists:
        ta_2 = label_topics2(string_list, ref2, 1, wiki_labels)
        topics_all_2.append(ta_2)
    return topics_all_2

#Reference word_list in list of strings
ref2 = list_of_strings(ref)
ta_2 = topics_all_2(topic_texts)
final_2 = final_tags(doc_tops, ta_2)


# ### Evaluation

def evaluation(y_lda, y):
    counter = 0
    for item in range(len(y_lda)):
        if y_lda[item][1] in y[item]:
            counter += 1
        else:
            continue
    return (counter/len(y_lda))

def accuracy_values(ends, y):
    accuracy_values = []
    for end in ends:
        acc = evaluation(end, y)
        accuracy_values.append(acc)
    return accuracy_values        

av = accuracy_values(final, y)
av_2 = accuracy_values(final_2, y)

#Source (4): https://stackoverflow.com/questions/7267226/range-for-floats

def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

font = font_manager.FontProperties(family='Times New Roman',size=12)
linestyle = 'dotted'

# Line plot: Prediction accuracy
limit=411; start=10; step=40;
x = range(start, limit, step)
y_axes = list(drange(0, 0.5, '0.05'))
x2 = range(start, limit, step)
y_axes_2 = list(drange(0, 0.5, '0.05'))
plt.plot(x, av, color = "royalblue")
plt.plot(x2, av_2, color = "green", linestyle=linestyle)
plt.xticks(x, **csfont)
plt.yticks(y_axes, **csfont)
plt.xlabel("Num Topics",**csfont)
plt.ylabel("Accuracy", **csfont)
plt.legend(('BOW', 'MPD'), prop = font)
plt.tight_layout()
#plt.show()
plt.savefig(wd+'/visuals/accuracy.png', dpi = 500)

# Model with highest accuracy
def best_performing_model(accuracy1, accuracy2):
    a1 = np.array(accuracy1)
    a2 = np.array(accuracy2)    
    m1 = a1.argmax()
    m2 = a2.argmax()   
    if a1[m1] > a2[m2]:
        print("Approach 1")
        print("Max accuracy: ", round(a1[m1],2) )
        return m1
    else:
        print("Approach 2")
        print("Max accuracy: ", round(a2[m2],2) )
        return m2      

bm = best_performing_model(av, av_2)
print('The model with the highest accuracy is the LDA specification: ', bm+1)

# P-test for accuracies

def load_accuracies(path):
    acc_BOW = []
    for file in glob.glob(path):
        proxy = open(file).read()
        proxy = proxy.replace('\n', " ")
        new = proxy.split(" ")
        tmp = []
        for item in new:
            z = item.strip(',][')
            tmp.append(float(z))
            l = np.array(tmp)
        acc_BOW.append(l)
    return np.array(acc_BOW)

accuracy_BOW = load_accuracies(wd+'/accuracies_BOW/*')
accuracy_MPD = load_accuracies(wd+'/accuracies_MPD/*')

# Significant difference between groups
ttest = stats.ttest_ind(accuracy_BOW,accuracy_MPD)
print('p-value: ', ttest.pvalue.mean())

# Significant trend within group
ttest_2 = stats.ttest_1samp(accuracy_BOW,0.2524597413125991)
print('p-value: ', ttest_2.pvalue[0])

# Frequency Table of predicted labels 
# (FROM HERE ONLY MOST ACCURATE MODEL)

def remove_index(final):
    new = []
    for l in final:
        tmp = []
        tmp.append(l[1])
        new.append(tmp)
    return new

# Remove indices from the final output
final_tags = remove_index(final[bm])

# Predicted distribution
# Frequency table and list of all topics predicted
unique_tags_result, dic_result, ft_result = frequency_table(final_tags)
ft_result1 = deepcopy(ft_result)
ft_result2 = deepcopy(ft_result)

# Most predicted labels
rev_dic_result = reverse_dic(dic_result)
frequency_result, most_used_result = most_freq_tags(ft_result1, rev_dic_result, 10)
print(most_used_result)
print(frequency_result)

# Compare most predicted labels to their actual distribution

#Actual distribution
actual_freq = []
for label in most_used_result:
    index = dic_result[label]
    frequency = ft_result2[index]
    actual_freq.append(frequency)
    
# Source (5): https://pythonspot.com/matplotlib-bar-chart/
# Barplot: Frequency of most predicted tag compared to its actual frequency
n_groups = 10
means_frank = frequency_result
means_guido = actual_freq
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='darkblue',
label='Predicted')
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='lightskyblue',
label='Actual')
plt.xlabel('Topics', **csfont )
plt.ylabel('Frequency', **csfont )
plt.xticks(index + bar_width, most_used_result, rotation=90, **csfont )
plt.yticks(**csfont )
plt.legend(prop= font)
plt.tight_layout()
plt.savefig(wd+'/visuals/frequency_actvspred.png')
#plt.show()

# Semantic Analysis 'technology'
# Find all documents flagged with tag of interest
def find_documents_original(y, tag):
    collection = []
    for i,j in enumerate(y):
        if tag in j:
            collection.append(i)
        else:
            continue
    return collection 

#Find all tags, the tag of interest is normally aligned with
def tags_in_common(collection, tag, y):
    c = []
    for index in collection:
        tmp = y[index]
        for item in tmp:
            if item == tag:
                continue
            else:
                if item in c:
                    continue
                else:
                    c.append(item)
    return c

'''Function compares the originally tagged documents to the actually tagged documents and returns 
    a list "tag-list" including all tags, that have been used instead of technology. Further, it returns a 
    frequency list of the topics used instead of technology and a dictionary to encode the frequency table
'''
def compare_original_predicted(final, original_tags_docs, y, tag_of_interest):
    actual_wrongly = []
    pred_wrongly = []
    unique = []
    pred_tech = 0
    pred_other = 0
    for item in original_tags_docs:
        #It doesn't necessarily mean that if a tag has not been predicted as "technology", it is accurate as it may
        #have just been predicted to be any of the other tags in y
        if final[item][1] not in y[item]:
            pred_wrongly.append(final[item][1])
            actual_wrongly.append(y[item])
            if final[item][1] in unique:
                continue
            else:
                unique.append(final[item][1])
        else:
            if final[item][1] == tag_of_interest:
                pred_tech += 1
            else:
                pred_other += 1
    #Create dictionary
    dic = {}
    index = 0
    for i in unique:
        dic[i] = index
        index += 1   
    #Create frequency table
    freq = np.zeros((len(dic)))
    for tag in pred_wrongly:
        freq[dic[tag]] +=1
    labels = list(dic.keys())
    return pred_tech, pred_other, pred_wrongly, actual_wrongly, dic, np.round(freq), labels

def freq_actual(y, unique, documents, tag):
    dic = {}
    index = 0
    for item in unique:
        if item not in dic:
            if item == tag:
                continue
            else:
                dic[item] = index
                index +=1   
    f = np.zeros(len(unique))
    for item in documents:
        tmp = y[item]
        for i in tmp:
            if i == tag:
                continue
            else:
                j = dic[i]
                f[j] += 1
    return f          

# List of indexes that represent of documents that are actually tagged with the label 'technology'
documents_technology = find_documents_original(y, "technology")

tags_aligned_technology = tags_in_common(documents_technology, "technology", y)

# Frequency table of how many times certain labels occur with 'technology'
freq_actual = freq_actual(y, tags_aligned_technology, documents_technology, "technology")

# Frequency table of wrongly predicted topics that should have been flagged with technology
ft_result3 = deepcopy(ft_result)
n_p, n_o, pred_label, actual_labels, dic, freq_pred, labels = compare_original_predicted(final[bm], documents_technology, y, 'technology')

def top_ten_tech(freq_pred, labels):
    labels_pred_tech = []
    freq_tech = []
    for item in range(10):
        index = freq_pred.argmax()
        labels_pred_tech.append(labels[index])
        freq_tech.append(freq_pred[index])
        freq_pred[index]=0
    return labels_pred_tech, freq_tech

labels_pred_tech, freq_tech = top_ten_tech(freq_pred, labels)
print(labels_pred_tech)
print(freq_tech)

# Calculate similarity (WP measure) between technology and the topics it has been wrongly identified with
similarities = []
w1 = wordnet.synset('technology.n.01')
w2 = wordnet.synset('sustainability.n.01')
w3 = wordnet.synset('computer.n.01')
w4 = wordnet.synset('hack.n.01')
w5 = wordnet.synset('friendship.n.01')
w6 = wordnet.synset('internet.n.01')
w7 = wordnet.synset('planet.n.01')
w8 = wordnet.synset('poverty.n.01')
w9 = wordnet.synset('money.n.01')
w10 = wordnet.synset('solar_energy.n.01')
w11 = wordnet.synset('film.n.01')
similarities.append(w1.wup_similarity(w2))
similarities.append(w1.wup_similarity(w3))
similarities.append(w1.wup_similarity(w4))
similarities.append(w1.wup_similarity(w5))
similarities.append(w1.wup_similarity(w6))
similarities.append(w1.wup_similarity(w7))
similarities.append(w1.wup_similarity(w8))
similarities.append(w1.wup_similarity(w9))
similarities.append(w1.wup_similarity(w10))
similarities.append(w1.wup_similarity(w11))

print(similarities)

#Get the words that are most similar to 'technology' based on WP similarity measure
WNL = WordNetLemmatizer()
new_unique_tags = []
for item in unique_tags:
    w = WNL.lemmatize(item)
    new_unique_tags.append(w)

nut = ['printer', 'activism', 'addiction', 'adventure', 'advertising', 'africa', 'age', 'agriculture', 'artificial_intelligence', 
       'aids', 'aircraft', 'algorithm', 'alternative_energy', 'alzheimers', 'ancient', 'animal', 'animation', 'anthropology', 
       'ant', 'ape', 'archaeology', 'architecture', 'art', 'asia', 'asteroid', 'astrobiology', 'astronomy', 'atheism', 'ar', 
       'autism', 'bacteria', 'beauty', 'bee', 'economics', 'problem', 'biodiversity', 'bioethics', 'biology', 'mechanics', 
       'biosphere', 'biotech', 'bird', 'blindness', 'body_language', 'book', 'botany', 'brain', 'brand', 'brazil', 'buddhism', 
       'bully', 'business', 'cancer', 'capitalism', 'car', 'cello', 'chemistry', 'child', 'china', 'choice', 'christianity', 'city', 
       'climate_change', 'cloud', 'code', 'cognitive_science', 'collaboration', 'comedy', 'communication', 'community', 'compassion', 
       'complexity', 'composing', 'computer', 'conducting', 'consciousness', 'conservation', 'consumerism', 'corruption', 'cosmos', 
       'creativity', 'crime', 'justice', 'culture', 'curiosity', 'dance', 'data', 'death', 'debate', 'decision', 'demo', 'democracy', 
       'depression', 'design', 'dinosaur', 'disability', 'disaster', 'discovery', 'disease', 'dna', 'drone', 'ebola', 
       'ecology', 'economics', 'education', 'egypt', 'empathy', 'energy', 'engineering', 'entertainment', 'entrepreneur', 'environment', 
       'epidemiology', 'europe', 'evil', 'evolution', 'exoskeleton', 'exploration', 'sport', 'failure', 'faith', 'family', 
       'fashion', 'fear', 'feminism', 'film', 'finance', 'fish', 'flight', 'food', 'forensics', 'friendship', 'future', 'gaming', 
       'garden', 'gender', 'equality', 'genetics', 'geology', 'glacier', 'god', 'google', 'government', 'grammar', 'green', 
       'guitar', 'gun', 'hack', 'happiness', 'health', 'health_care', 'hearing', 'history', 'hiv', 'human', 'humanity', 'humor',
       'identity', 'illusion', 'immigration', 'india', 'inequality', 'infrastructure', 'innovation', 'insect', 'intelligence', 
       'internet', 'interview', 'introvert', 'invention', 'investment', 'iran', 'iraq', 'islam', 'jazz', 'journalism', 'language', 
       'law', 'leadership', 'library', 'life', 'literature', 'music', 'love', 'magic', 'manufacturing', 'map', 'marketing', 
       'mar', 'material', 'math', 'medium', 'medicine', 'meditation', 'meme', 'memory', 'men', 'mental_health', 'microbe',
       'microbiology', 'middle_east', 'military', 'mind', 'mindfulness', 'mining', 'ocean', 'mobility', 'money', 
       'monkey', 'moon', 'morality', 'motivation', 'movie', 'museum', 'music', 'narcotic', 'nasa', 'resource', 'nature', 
       'neuroscience', 'new_york', 'news', 'nobel_prize', 'novel', 'nuclear_energy', 'nuclear_weapon', 'obesity', 'oil', 
       'origami', 'pain', 'painting', 'paleontology', 'pandemic', 'parent', 'peace', 'performance', 'personality',
       'pharmaceutical', 'philanthropy', 'philosophy', 'photography', 'physic', 'physiology', 'piano', 'planet', 'plant', 
       'plastic', 'play', 'poetry', 'policy', 'politics', 'pollution', 'population', 'potential', 'poverty', 'prediction', 
       'pregnancy', 'presentation', 'primate', 'prison', 'privacy', 'productivity', 'programming', 'prosthetics', 'protest', 
       'psychology', 'race', 'refugee', 'relationship', 'religion', 'river', 'robot', 'rocket', 'sanitation', 'science', 
       'security', 'self', 'sens', 'sex', 'shopping', 'sight', 'simplicity', 'singer', 'skateboard', 'slavery', 'sleep', 'smell', 
       'society', 'sociology', 'software', 'solar_energy', 'solar_system', 'sound', 'south_america', 'space', 'speech', 'word',
       'statistic', 'student', 'submarine', 'success', 'suicide', 'surgery', 'surveillance', 'sustainability', 'syria', 
       'teaching', 'telecom', 'telescope', 'television', 'terrorism', 'theater', 'time', 'toy', 
       'transportation', 'travel', 'tree', 'trust', 'typography', 'united_states', 'universe', 
       'vaccine', 'violence', 'violin', 'virtual_reality', 'virus', 'visualization', 'vulnerability', 'war', 'water', 
       'weather', 'web', 'woman', 'work', 'writing', 'youth']      

s = []
for item in nut:
    wX = wordnet.synset(item+'.n.01')
    s.append(w1.wup_similarity(wX))

s = np.array(s)
s2 = deepcopy(s)
s3 = deepcopy(s)

# Find the four most and the four least similar words to 'technology'
def most_sim(nut, similarity, num=4):
    s2 = similarity[:]
    label_max = []
    simi_max = []
    for counter in range(num):
        index = s2.argmax()
        simi_max.append(s2[index])
        label_max.append(nut[index])
        s2[index] = 0
    return label_max, simi_max

def least_sim(nut, similarity, num=4):
    s3 = similarity[:]
    label_min = []
    simi_min = []
    for counter in range(num):
        index = s3.argmin()
        simi_min.append(s3[index])
        label_min.append(nut[index])
        s3[index] = 100
    return label_min, simi_min

label_max, simi_max = most_sim(nut, s2, 4)
print(label_max, simi_max)

label_min, simi_min = least_sim(nut, s3, 4)
print(label_min, simi_min)

# Semantic similarity on document level
n = random.randint(1,712)
print(n)

print(documents_technology[517])

# For example document
print(final[bm][1614])
print(y[1614])

# Similarity analysis random document 395
o = ['activism', 'design', 'fashion', 'future', 'prosthetics', 'sport', 'technology']
w_6 = wordnet.synset('friendship.n.01')
s_395 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_395.append(w_6.wup_similarity(wX))
print(s_395)

# Similarity analysis random document 431
o = ['vaccine', 'business', 'ebola', 'health', 'health_care', 'history', 'technology']
w_1 = wordnet.synset('disease.n.01')
s_431 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_431.append(w_1.wup_similarity(wX))
print(s_431)

# Similarity analysis random document 1574
o = ['book', 'computer', 'history', 'library', 'map', 'technology']
w_2 = wordnet.synset('science.n.01')
s_1574 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_1574.append(w_2.wup_similarity(wX))
print(s_1574)

# Similarity analysis random document 1444
o = ['drone', 'entertainment', 'robot', 'technology', 'war', 'writing']
w_3 = wordnet.synset('government.n.01')
s_1444 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_1444.append(w_3.wup_similarity(wX))
print(s_1444)

# Similarity analysis random document 273
o = ['entertainment', 'music', 'performance', 'piano', 'technology']
w_4 = wordnet.synset('sound.n.01')
s_273 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_273.append(w_4.wup_similarity(wX))
print(s_273)

# Similarity analysis random document 500
o = ['entertainment', 'music', 'performance', 'piano', 'technology']
w_5 = wordnet.synset('sound.n.01')
s_500 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_500.append(w_5.wup_similarity(wX))
print(s_500)

# Similarity analysis random document 395
o = ['activism', 'design', 'fashion', 'future', 'prosthetics', 'sport', 'technology']
w_6 = wordnet.synset('friendship.n.01')
s_395 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_395.append(w_6.wup_similarity(wX))
print(s_395)

# Similarity analysis random document 42
o = ['culture', 'genetics', 'science', 'statistics', 'technology']
w_7 = wordnet.synset('morality.n.01')
s_42 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_42.append(w_7.wup_similarity(wX))
print(s_42)

# Similarity analysis random document 498
o = ['design', 'entertainment', 'material', 'technology']
w_8 = wordnet.synset('computer.n.01')
s_498 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_498.append(w_8.wup_similarity(wX))
print(s_498)

# Similarity analysis random document 569
o = ['adventure', 'aircraft', 'engineering', 'green', 'solar_energy', 'technology']
w_9 = wordnet.synset('flight.n.01')
s_569 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_569.append(w_9.wup_similarity(wX))
print(s_569)

# Similarity analysis random document 1614
o = ['disability', 'language', 'speech', 'technology']
w_10 = wordnet.synset('consciousness.n.01')
s_1614 = []
for item in o:
    wX = wordnet.synset(item+'.n.01')
    s_1614.append(w_10.wup_similarity(wX))
print(s_1614)


# ### External Sources

# 1. https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python/14598135
# 2. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# 3. Inspired from code used in course: Data Processing Advanced, Notebook 'Distance and Similarity', D.Hendrickson, Tilburg University
# 4. https://stackoverflow.com/questions/7267226/range-for-floats
# 5. https://pythonspot.com/matplotlib-bar-chart/
