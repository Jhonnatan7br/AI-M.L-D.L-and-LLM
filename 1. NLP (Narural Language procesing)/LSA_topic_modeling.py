"""LSA Model: Prepare requirements described on README.md file, Topic Modeling Gensim file and official documentation disposed throught this project"""

#%%
LDA_documentation = 'https://radimrehurek.com/gensim/models/ldamodel.html'

Key_concepts = [
#Document: some text.
#Corpus: a collection of documents.
#Vector: a mathematically convenient representation of a document.
#Model: an algorithm for transforming vectors from one representation to another.
]

import pandas as pd
import pprint
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel
from gensim.test.utils import common_texts


# Read your CSV file
research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")

# Create a sub-dataset with the first 10 lines
sub_dataset = research.head(5251)

# Extract descriptions from the 'Abstract' column of the dataframe
text_corpus = sub_dataset['Abstract'].tolist()
#%%
"""Preprocess your text data as you did before (lowercase, remove stopwords, etc.)"""
stop_words = set(nlp.Defaults.stop_words)
new_stop_words = {'ai','(ai)','patien','patients','=','de','control','model','system','et','results','power','±','new','compared','risk','data','smart','la','abstract','om','“no','available','set','problem','teatures','siven','class','rights','general','cows','milk','relevant','reserved."','time','pattern','constraint','classiters','vve I','problems','consider','propositional','logic','present','space','large','springer','prove','intake','la','©','found','problem','features','given','p','(p)','knowledge','results','si','d','"[no','available]"','use','field','"The','provide','based','paper','propose','decision','2021','process','methods','paper,','however','number','studies','study','<','conclusion:','2','1','significant','included','total','(n','des','les','genetic','à','en','le','une','qui','du','associated','literature','review',',','intelligence','artificial', 'approach','proposed','intelligence','accuracy','parameters','group','methods:','results:','(p','low', 'different', 'higher', 'analysis', 'published', 'articles','coding', 'dans', 'un', 'video', 'se', 'que', 'productivity', 'pour', 'elsevier','automatic','levels', 'expression', 'increased', 'effect', 'days', 'high','however','index', 'significantly', 'af', 'function'}
stop_words.update(new_stop_words)

# Tokenize and preprocess your text data
texts = [[word for word in document.lower().split() if word not in stop_words] for document in text_corpus]

# Create a dictionary and a bag-of-words corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

num_topics = 10
# Create an LSA model
lsa_model = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)  # Adjust the number of topics as needed

# Print the topics
pprint.pprint(lsa_model.print_topics())

#%%
"""Get the topics from the LSI model"""

lsi_topics = lsa_model.show_topics(num_topics=num_topics, formatted=False)

# Print the topics in human-readable format
for topic_id, topic in lsi_topics:
    words = [word for word, _ in topic]
    topic_text = ', '.join(words)
    print(f"Topic {topic_id}: {topic_text}")