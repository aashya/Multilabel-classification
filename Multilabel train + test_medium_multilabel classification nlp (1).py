#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
import re


# In[2]:


train_df = pd.read_csv('C:/Users/aashya.khanduja/Documents/Aashya/Xane/SET1_Theming_data.csv')
test_df = pd.read_csv('C:/Users/aashya.khanduja/Documents/Aashya/Xane/Test.csv')


# In[3]:


train_df.sample(5)


# In[4]:


def removenumbers(train_df):
#Remove numbers
    train_df = re.sub(r'\d+', '', str(train_df))
    #print(train_df)
    return train_df
   


# In[5]:


def convertlower(train_df):
    #print(train_df)
    train_df = train_df.lower();
    return train_df
#print(train_df)


# In[6]:


#deleting symbols etc
def removepunctuation(train_df):
    import string 
    table = str.maketrans("","",string.punctuation);
    train_df=train_df.translate(table);
    return train_df
#def remove_punctuation(text):
#      return text.translate(table)
#train_df['new'] = train_df['review'].apply(removing_punct)
#train_df.set(i for i in text if i in string.punctuation)
    #translate(str.maketrans("",""), str.punctuation)
#print(train_df)


# In[7]:


def strip(train_df):
    train_df = train_df.strip()
    return train_df
#train_df


# In[8]:


def preProcessing(train_df):
    train_df = removenumbers(train_df)
    train_df = convertlower(train_df)
    train_df = removepunctuation(train_df)
    train_df = strip(train_df)
    return train_df


# In[9]:


totalText='' 
for x in train_df['review']:
    #print(x)
    #print(type(x), x)
    ps=preProcessing(x)
    #print (train_df)
    totalText = totalText+""+ps 
#    from wordcloud import WordCloud 
#    wc=WordCloud(background_color='black',max_font_size=50).generate(totalText) 
#     plt.figure(figsize=(16,12)) 
#     plt.imshow(wc, interpolation="bilinear")


# In[10]:


import nltk 
from nltk.tokenize import ToktokTokenizer 
x=nltk.FreqDist(ToktokTokenizer().tokenize(totalText)) 
print(type(x))
#plt.figure(figsize=(16,5)) 
#x.plot(20)


# In[11]:


from sklearn.datasets import make_multilabel_classification


# In[16]:


y = make_multilabel_classification(sparse = True, n_labels = 20, return_indicator = 'sparse', allow_unlabeled = False)
print(type(y))


# In[17]:


import pandas as dataframe
from skmultilearn.problem_transform import BinaryRelevance 
from sklearn.naive_bayes import GaussianNB


# In[18]:


classifier = BinaryRelevance(GaussianNB())


# In[20]:


classifier.fit(x.astype(tuple),y)


# In[ ]:


predictions = classifier.predict(X)


# In[ ]:


#train_df.head(10)


# In[ ]:


#print(train_df['review'])


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(train_df, predictions))


# In[ ]:


cols_target = ['Colleague Relations', 'Company Feedback','Communication','Job Satisfaction','Leadership','Others','Personal Growth','Recognition','Relationship with Manager','Resources & Benefits','Training & Process Orientation','Vision Alignment']


# In[ ]:





# In[ ]:


train_df.describe()


# In[ ]:




