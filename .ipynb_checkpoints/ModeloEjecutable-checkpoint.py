#!/usr/bin/env python
# coding: utf-8

# ## 1. Importación de librerias

# In[86]:


# Instalación de librerias
# librería Natural Language Toolkit, usada para trabajar con textos 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

import re, unicodedata, string
import contractions
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, get_scorer_names, f1_score, make_scorer, classification_report
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt


# ## 2. Cargar datos iniciales

# In[104]:


def cargaDeFramework():
    frameworkData = pd.read_csv('data/verifiedArticles.csv', sep=',', encoding = 'ANSI')
    return frameworkData


# In[100]:


def cargaDeEntrada():    
    inputData = pd.read_csv('data/dataToClassify.csv', sep=',', encoding = 'utf-8')
    return inputData


# ## 3. Preparación de datos

# In[89]:


def eliminarDuplicados(dataframe):
    return dataframe.drop_duplicates()


# In[90]:


stop_words = stopwords.words('english')
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def remove_punctuation_and_numbers(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        word = word.strip()  
        word = re.compile('<.*?>').sub('', word) 
        word = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', word)  
        word = re.sub('\s+', ' ', word)  
        word = re.sub(r'\[[0-9]*\]',' ', word) 
        word = re.sub(r'[^\w\s]', '', str(word).lower().strip())
        word = re.sub(r'\d',' ', word) 
        word = re.sub(r'\s+',' ', word) 
        if word != "":
            new_words.append(word)
    return new_words
        
    
def preprocessing(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_numbers(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


# In[91]:


lemmatizer = WordNetLemmatizer()
def lemmatize_verbs(words):
    new_words = []
    for word in words:
        new_word = lemmatizer.lemmatize(word)
        new_words.append(new_word)
    return new_words


# In[92]:


def tokenizacionLematizacion(dataframe):
    dataframe["Abstract"] = dataframe["Abstract"].apply(contractions.fix)
    dataframe['Words'] = dataframe['Abstract'].apply(word_tokenize).apply(preprocessing)
    dataframe['Words'] = dataframe['Words'].apply(lemmatize_verbs)
    dataframe['Words'] = dataframe['Words'].apply(lambda x: ' '.join(map(str, x)))
    return dataframe


# In[93]:


def eliminacionDeStopwords(dataframe, dataframeStopwords):
    dataframe["Words"] = dataframe["Words"].replace('|'.join(dataframeStopwords), '', regex=True)
    dataframe["Words"] = dataframe["Words"].replace(value='', regex=r'\b[a-z]{1,2}\b')
    return dataframe


# In[106]:


def transformacionDeFramework(frameworkData):
    frameworkData = frameworkData[frameworkData["Category"].notna()]
    frameworkData = frameworkData[["Title", "Abstract", "Category"]]
    frameworkData['Abstract'] = frameworkData['Abstract'] + " " + frameworkData['Title']
    tokenizacionLematizacion(frameworkData)
    frameworkStopwords = ["project", "area", "given", "level", "buzios", "world", "contain", "best", "within", "field", "paper", "around", "public", "ability", "making"
                      "develop", "purpose", "using", "nature", "present", "author", "concept", "number", "proposed", "result", "contain", "different", "several",
                      "management", "portfolio", "focus", "help", "however", "term", "problem", "time", "many", "system", "case", "process", "make", "set", "use", 
                      "give", "lean", "open", "well", "key", "oil", "also", "new", "include", "single", "face", "rapid", "long", "built", "follow", "consequently",
                      "today", "achieve", "realize", "developed", "public", "constantly", "one", "identify", "give", "need", "several", "often", "show", "become",
                      "although", "aim", "manage", "non", "site", "pre", "vital", "responibility", "applicable"]
    eliminacionDeStopwords(frameworkData, frameworkStopwords)
    frameworkData.to_csv('data/transformedFrameworkData.csv')
    return frameworkData


# In[107]:


def transformacionDeEntrada(inputData):
    inputData = inputData[["Title", "Abstract", "Author Keywords", "Index Keywords"]]
    inputData['Author Keywords'] = inputData['Author Keywords'].fillna("")
    inputData['Index Keywords'] = inputData['Index Keywords'].fillna("")
    inputData['Abstract'] = inputData['Abstract'] + " " + inputData['Title'] + " " + inputData['Author Keywords'] + " " +  inputData['Index Keywords']
    inputData = inputData.drop(columns = ['Index Keywords', 'Author Keywords'])
    tokenizacionLematizacion(inputData)
    inputStopwords = ["project", "portfolio", "using", "develop", "approach", "system", "tool", "used", "team", "current", "activity", "structure", "present", "data",
                   "need", "within", "open", "right", "time", "paper", "proceeding", "new", "different", "towards", "case", "topic", "based", "set", "use", "give",
                   "make", "need", "purpose", "manage", "new", "show", "aim"]
    eliminacionDeStopwords(inputData, inputStopwords)
    inputData.to_csv('data/transformedInputData.csv')
    return inputData


# ## 4. Preparación de conjunto de datos preentrenados

# In[102]:


numFeatures = 0
tfidfconverterF = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 1), encoding='latin-1', min_df=3, max_df=0.4, stop_words=('english'))
def crearModeloFramework(frameworkData):
    featuresF = tfidfconverterF.fit_transform(frameworkData.Words).toarray()
    labels = frameworkData['Category']
    smote = SMOTE(random_state=0, k_neighbors=2)
    featuresSmote, labelsSmote = smote.fit_resample(featuresF, labels)
    numFeatures = featuresSmote.shape[1]
    model = OneVsRestClassifier(LinearSVC())
    model.fit(featuresSmote, labelsSmote);
    return model


# ## 5. Predicción de etiquetas y actualización de datos entrantes

# In[97]:


def predecirCategorias(inputData, model):
    featuresS = tfidfconverterF.transform(inputData.Words).toarray()
    inputData["Category"] = model.predict(featuresS)
    return inputData


# ## 6. Carga de predicción de datos

# In[98]:


def cargarPrediccion(inputData):
    inputData = inputData.drop(columns = ['Words'])
    inputData.to_csv('data/classifiedData.csv')


# ## 7. Ejecución de pipeline

# In[108]:


frameworkData = (
    cargaDeFramework()
    .pipe(eliminarDuplicados)
    .pipe(transformacionDeFramework)
)
modeloLinearSVC = crearModeloFramework(frameworkData)
inputData = (
    cargaDeEntrada()
    .pipe(eliminarDuplicados)
    .pipe(transformacionDeEntrada)
    .pipe(predecirCategorias, model = modeloLinearSVC)
    .pipe(cargarPrediccion)
)


# In[ ]:




