#!/usr/bin/env python
# coding: utf-8

# In[66]:


# Importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# Data Uploading
data = pd.read_csv('UpdatedResumeDataSet.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data['Cleaned_Resume'] = ''


# In[6]:


data.head()


# In[7]:


# Checking Unique Category
print(data['Category'].unique())


# In[8]:


# Counting Different Categories
print(data['Category'].value_counts())


# In[9]:


# Checking Null Values 
data.isnull().sum()


# In[10]:


# Visualizing Categorize
plt.figure(figsize = (15,15))
plt.xticks(rotation=90)
sns.countplot( y = 'Category', data=data)


# In[11]:


# Showing GridSpec 
from matplotlib.gridspec import GridSpec

targetCounts = data.Category.value_counts()
targetLabels = data.Category.unique()

plt.figure(1, figsize=(25,55))
the_gird = GridSpec(2,2)

cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0,1,3)]
plt.subplot(the_gird[0,1], aspect = 1, title = 'Category Distribution')

source_pie = plt.pie(targetCounts, labels = targetLabels, autopct = '%1.1f%%', shadow = True, colors= colors)
plt.show()


# In[12]:


# Using Regular expressions to remove URL, Hashtags, mentions special letters,and punctuations
import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
    


# In[13]:


data['Cleaned_Resume'] = data['Resume'].apply(lambda x: cleanResume(x))


# In[14]:


data.head()


# In[41]:


# Word Distribution and Word Cloud

import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = data['Resume'].values
cleanedSentences = ""
for i in range(0,160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure( figsize = (15,15))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[54]:


# Converting Words to Categorical Values:
from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']

le = LabelEncoder()


for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[62]:


# Training Model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = data['Cleaned_Resume'].values 
requiredTarget = data['Category'].values

word_verctorizer = TfidfVectorizer (
    sublinear_tf= True,
    stop_words= 'english',
    max_features = 1500
)

word_verctorizer.fit(requiredText)

WordFeatures = word_verctorizer.transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state =0 , test_size = 0.2 )
print(X_train.shape)
print(X_test.shape)


# In[76]:


# Training Model Continues

clf = OneVsRestClassifier(KNeighborsClassifier())

clf.fit(X_train,y_train)

prediction = clf.predict(X_test)

print("Accuracy of K Neihbors Classfier on training Set: {:.2f}".format(clf.score(X_train,y_train)))
print("Accuracy of K Neihbors Classfier on test Set: {:.2f}".format(clf.score(X_test,y_test)))


print("Classfication Report for Classfier  %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))




# In[77]:





# In[ ]:




