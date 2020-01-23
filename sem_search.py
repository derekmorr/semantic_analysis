import argparse
import numpy as np
import pdf_convert
import os
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import tensorflow_hub as hub
import Timer as t

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

parser.add_argument(
    '--pdf_embed_dir',
    type=str,
    default='',
    help='Path to PDF embedding file'
)


FLAGS, unparsed = parser.parse_known_args()

timer_total = t.Timer()

print(f'Total start is {timer_total}.')

sentences = pdf_convert.convert(FLAGS.pdf_dir)

tf_sentences = tf.reshape(sentences, [-1])
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)
timer_e = t.Timer()

embeddings = embed(
            tf_sentences,
            signature="default",
            as_dict=True)["default"]

search_string = FLAGS.search_string 
print("SEARCH: " + search_string)
results_returned = str(FLAGS.num_results)

timer_x = t.Timer()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  try:
    x = np.load(FLAGS.pdf_embed_dir)
  except: 
    x = sess.run(embeddings)
    np.save(FLAGS.pdf_embed_dir, x)

from sklearn.metrics.pairwise import cosine_similarity

embeddings2 = embed(
    [search_string],
    signature="default",
    as_dict=True)["default"]

timer_sv = t.Timer()
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
  
print(f'Total end is {timer_total}.')