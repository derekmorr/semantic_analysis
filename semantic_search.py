from io import StringIO
import logging
import os

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import spacy
from sklearn.metrics.pairwise import cosine_similarity

def convert(fname, pages=None):
    "converts pdf, returns its text content as a string"
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
    output.close()
    return text

COMPENDIUM_FOLDER_PATH = "/Users/singhcpt/dev/semantic_analysis/test_compendium"

pdf_texts = []

for filename in os.listdir(COMPENDIUM_FOLDER_PATH):
    if ".pdf" in filename:
        pdf_texts.append(convert(COMPENDIUM_FOLDER_PATH + "/" + filename))

# python -m spacy download en_core_web_md you will need to install this on first load
nlp = spacy.load('en_core_web_md')
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

text = pdf_texts[0].lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')
text = ' '.join(text.split())
doc = nlp(text)

sentences = []
for i in doc.sents:
    if len(i) > 1:
        sentences.append(i.string.strip())

tf_sentences = tf.reshape(sentences[0:1000], [-1])

embeddings = embed(
    tf_sentences,
    signature="default",
    as_dict=True)["default"]
search_string = "what should i do if my tomato has late blight" #@param {type:"string"}
print("SEARCH: " + search_string)
results_returned = "10"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    x = sess.run(embeddings)

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

print("RESULTS:")

x = 0
for i, j in cosine_similarities.nlargest(int(results_returned)).iteritems():
    x += 1
    print("RESULT " + str(x) + ": " + sentences[i] + "\n")
