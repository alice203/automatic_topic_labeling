import glob
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wd = os.getcwd()

#Load stm_scripts
def load_scripts(path):
    strings = []
    for seq_path in sorted(glob.glob(path)):
        #print(seq_path)
        proxy = open(seq_path).read()
        proxy = proxy.replace('\n', " ")
        strings.append(proxy)
    return strings

stm = load_scripts(wd+"/stm/*.stm")

def clean_strings(strings):
    l = []
    for string in strings:
        final = ""
        e = string.split(" ")[0]
        start = -1
        while True:
            start = string.find( "<NA>", start+1 )
            if start < 0:
                break
            finish = string.find( e , start )
            if finish < 0:
                break
            final += string[start+5:finish]
            start = finish

        final = final.replace("<unk> ","")
        final = final.replace("  "," ")
        
        #To be
        final = final.replace("i 'm", "i am")
        final = final.replace("you 're", "you are")
        final = final.replace("he 's", "he is")
        final = final.replace("she 's", "she is")
        final = final.replace("it 's", "it is")
        final = final.replace("we 're", "we are")
        final = final.replace("they 're", "they are")
        final = final.replace("it 's", "it is")
        
        #Auxilary words
        final = final.replace("i 'd", "i would")
        final = final.replace("you 'd", "you would")
        final = final.replace("he 'd", "he would")
        final = final.replace("she 'd", "she would")
        final = final.replace("we 'd", "we would")
        final = final.replace("they 'd", "they would")
        
        #To have
        final = final.replace("i 've", "i have")
        final = final.replace("you 've", "you have")
        final = final.replace("we 've", "we have")
        final = final.replace("they 've", "they have")
        
        #Future
        final = final.replace("i 'll", "i will")
        final = final.replace("you 'll", "you will")
        final = final.replace("we 'll", "we will")
        final = final.replace("they 'll", "they will")
        final = final.replace("it 'll", "it will")
        final = final.replace("he 'll", "he will")
        final = final.replace("she 'll", "she will")
        
        #Negative
        final = final.replace("didn 't", "did not")
        final = final.replace("don 't", "do not")
        final = final.replace("doesn 't", "does not")
        final = final.replace("wasn 't", "was not")
        final = final.replace("weren 't", "were not")
        final = final.replace("won 't", "will not")
        final = final.replace("wouldn 't", "would not")
        final = final.replace("shouldn 't", "should not")
        final = final.replace("couldn 't", "could not")
        final = final.replace("haven 't", "have not")
        final = final.replace("hasn 't", "has not")
        
        #Questions
        final = final.replace("what 's", "what is")
        final = final.replace("where 's", "where is")
        final = final.replace("who 's", "who is")
        final = final.replace("how 's", "how is")
        final = final.replace("why 's", "why is")
        
        #Others
        final = final.replace("that 's", "that is")
        final = final.replace("there 's", "there is")
        final = final.replace("here 's", "here is")
        final = final.replace("'","")
        
        final = final.strip().split(" ")
        l.append(final)
    return l

scriptlist = clean_strings(stm)