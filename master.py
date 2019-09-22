#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install sphfile


# In[4]:


from sphfile import SPHFile


# In[5]:


sph =SPHFile('/Users/aliciahorsch/Downloads/TEDLIUM_release-3/data/sph/AalaElKhani_2016X.sph')
# Note that the following loads the whole file into ram
print( sph.format )
# write out a wav file with content
sph.write_wav( 'AalaElKhani_2016X.wav' )


# In[ ]:





# In[14]:


file = open("ZubaidaBai_2016S.stm")
first = file.read()
word_list = first.split(" ")
#print(word_list)

new_list = []
for word in word_list:
    if word not in new_list:
        new_list.append(word)
    else:
        continue

#Can't do that because of word count        
#print(new_list)


# In[ ]:




