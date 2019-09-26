import argparse
import logging
import numpy as np
import os
import pandas as pd
import pdf_converter
from sklearn import preprocessing
import spacy
from spacy.lang.en import English
from spacy import displacy
import tensorflow as tf
import tensorflow_hub as hub

parser = argparse.ArgumentParser()

parser.add_argument(
    '--pdf_dir',
    type=str,
    default='',
    help='Path to folder of pdf files'
)

parser.add_argument(
    '--search_string',
    type=str,
    default='',
    help='Enter string for semantic search'
)

parser.add_argument(
    '--num_results',
    type=int,
    default='',
    help='Number of results to return'
)

FLAGS, unparsed = parser.parse_known_args()

pdf_texts = []

for filename in os.listdir(FLAGS.pdf_dir):
    if (".pdf" in filename):
      print("Converting " + filename + " to pdf.") 
      pdf_texts.append(pdf_converter.convert(FLAGS.pdf_dir + "/" + filename))

# python -m spacy download en_core_web_md you will need to install this on first load
nlp = spacy.load('en_core_web_md')
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

import re

# NEED TO CHANGE THIS LATER, INDEXING FIRST PDF ONLY
text = pdf_texts[0].lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
text = ' '.join(text.split())
doc = nlp(text)

sentences = []
for i in doc.sents:
  if len(i) > 1:
    sentences.append(i.string.strip())

# NEED TO CHANGE THIS LATER
tf_sentences = tf.reshape(sentences[0:1000], [-1])

embeddings = embed(
            tf_sentences,
            signature="default",
            as_dict=True)["default"]
search_string = FLAGS.search_string #@param {type:"string"}
print("SEARCH: " + search_string)
results_returned = str(FLAGS.num_results) #@param [1, 2, 3]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)

from sklearn.metrics.pairwise import cosine_similarity

embeddings2 = embed(
    [search_string],
    signature="default",
    as_dict=True)["default"]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  search_vect = sess.run(embeddings2)
  
cosine_similarities = pd.Series(cosine_similarity(search_vect, x).flatten())
output = ""
x = 0

print("\nRESULTS:\n")

for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
  x += 1
  print("RESULT " + str(x) + ": " + sentences[i] + "\n")