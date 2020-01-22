from io import StringIO
import sys

#sys.path.append("/usr/home/username/pdfminer")

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import getopt

#converts pdf, returns its text content as a string
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 

import os

COMPENDIUM_FOLDER_PATH = "/Users/singhcpt/dev/semantic_analysis/test_compendium"
# COMPENDIUM_FOLDER_PATH = "/Users/singhcpt/Documents/APS_Compendia"

pdf_texts = []

for filename in os.listdir(COMPENDIUM_FOLDER_PATH):
    if (".pdf" in filename):
      pdf_texts.append(convert(COMPENDIUM_FOLDER_PATH + "/" + filename))

# from here on is what we need on the VM
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing

# python -m spacy download en_core_web_md you will need to install this on first load
import spacy
from spacy.lang.en import English
from spacy import displacy
nlp = spacy.load('en_core_web_md')
from IPython.display import HTML
import logging
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

import re

text = pdf_texts[0].lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
text = ' '.join(text.split())
doc = nlp(text)

sentences = []
for i in doc.sents:
  if len(i)>1:
    sentences.append(i.string.strip())

tf_sentences = tf.reshape(sentences[0:1000], [-1])

embeddings = embed(
            tf_sentences,
            signature="default",
            as_dict=True)["default"]
search_string = "what should i do if my tomato has late blight" #@param {type:"string"}
print("SEARCH: " + search_string)
results_returned = "10" #@param [1, 2, 3]

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

print("RESULTS:")

for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
  x += 1
  print("RESULT " + str(x) + ": " + sentences[i] + "\n")