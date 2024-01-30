"""LDA Model: Prepare requirements described on README.md file, Topic Modeling Gensim file and official documentation disposed throught this project"""

#%%
LDA_documentation = 'https://radimrehurek.com/gensim/models/ldamodel.html'

Key_concepts = [
#Document: some text.
#Corpus: a collection of documents.
#Vector: a mathematically convenient representation of a document.
#Model: an algorithm for transforming vectors from one representation to another.
]

import pprint
import pandas as pd
from gensim import corpora, models
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

import spacy
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")
# Create a sub-dataset with the first 10 lines
sub_dataset = research.head(5251)
# Extract descriptions from the 'description' column of the dataframe
text_corpus = sub_dataset['Abstract'].tolist()
text_corpus = [f'"{doc}"' for doc in text_corpus]

# Create a set of stopwords (set of frequent words):
#stop_words = set(stopwords.words('english'))  # nltk stopwords in english for English 
stop_words = set(nlp.Defaults.stop_words)
# Add manually another stop word to reduce model noise
new_stop_words = {'ai','(ai)','patien','patients','=','de','control','model','system','et','results','power','±','new','compared','risk','data','smart','la','abstract','om','“no','available','set','problem','teatures','siven','class','rights','general','cows','milk','relevant','reserved."','time','pattern','constraint','classiters','vve I','problems','consider','propositional','logic','present','space','large','springer','prove','intake','la','©','found','problem','features','given','p','(p)','knowledge','results','si','d','"[no','available]"','use','field','"The','provide','based','paper','propose','decision','2021','process','methods','paper,','however','number','studies','study','<','conclusion:','2','1','significant','included','total','(n','des','les','genetic','à','en','le','une','qui','du','associated','literature','review',',','intelligence','artificial', 'approach','proposed','intelligence','accuracy','parameters','group','methods:','results:','(p','low', 'different', 'higher', 'analysis', 'published', 'articles','coding', 'dans', 'un', 'video', 'se', 'que', 'productivity', 'pour', 'elsevier','automatic','levels', 'expression', 'increased', 'effect', 'days', 'high','however','index', 'significantly', 'af', 'function'}
# Add the new words to the existing set
stop_words.update(new_stop_words)
# Building set of stop_words manually
#stoplist = set('for a of the and to in'.split(' '))

# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stop_words]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1 # Increase frecuency Token or decrease depending espected results

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)


# Attempting to directly pass processed_corpus to LdaModel without first converting it to a Bag-of-Words (BoW) format using the doc2bow method for each document

# Create Dictionary for processed corpus
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

# Convert the dictionary to a bag-of-words corpus for reference.
corpus = [dictionary.doc2bow(text) for text in processed_corpus]

""" Train an LDA model using a Gensim corpus """
# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
# Train the LDA model on the BoW corpus.
#lda = LdaModel(corpus, num_topics=10)
# Train the LDA model
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=30)
#lda = models.LdaModel(corpus, num_topics=10)

# Print LDA Topic Modeling vector
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
# Generate human-readable topic names
# Iterate through each topic
for topic_number, topic in lda_model.print_topics(num_topics=10, num_words=10):
    # Extract words and their weights
    words = topic.split('+')
    # Remove the weights and keep only the words
    words = [word.split('*')[1].replace('"', '').strip() for word in words]
    # Create a readable string for each topic
    topic_str = ", ".join(words)
    print(f"Topic {topic_number}: {topic_str}")

# Create a visualization
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)

#%%

# Print LDA Topic Modeling vector
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

#%%
    
# Generate human-readable topic names
# Iterate through each topic
for topic_number, topic in lda_model.print_topics(num_topics=10, num_words=10):
    # Extract words and their weights
    words = topic.split('+')
    # Remove the weights and keep only the words
    words = [word.split('*')[1].replace('"', '').strip() for word in words]
    # Create a readable string for each topic
    topic_str = ", ".join(words)
    print(f"Topic {topic_number}: {topic_str}")