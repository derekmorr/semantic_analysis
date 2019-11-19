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

FLAGS, unparsed = parser.parse_known_args()

timer_total = t.Timer()

print(f'Total start is {timer_total}.')

sentences = pdf_convert.convert(FLAGS.pdf_dir)

tf_sentences = tf.reshape(sentences, [-1])
os.environ['TFHUB_CACHE_DIR'] = '/Users/singhcpt/dev/semantic_analysis/tf_cache'
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)
timer_e = t.Timer()
print(f'Embedding start is {timer_e}.')
embeddings = embed(
            tf_sentences,
            signature="default",
            as_dict=True)["default"]
print(f'Embedding end is {timer_e}.')

search_string = FLAGS.search_string #@param {type:"string"}
print("SEARCH: " + search_string)
results_returned = str(FLAGS.num_results) #@param [1, 2, 3]

timer_x = t.Timer()
print(f'Sess 1 start is {timer_x}.')
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  # x = sess.run(embeddings)
  # print(f'Time elapsed is {timer_x}.')
  try:
    x = np.load('/Users/singhcpt/dev/semantic_analysis/x.npy')
  except: 
    x = sess.run(embeddings)
    np.save('/Users/singhcpt/dev/semantic_analysis/x.npy', x)

print(f'Sess 1 end is {timer_x}.')

from sklearn.metrics.pairwise import cosine_similarity

embeddings2 = embed(
    [search_string],
    signature="default",
    as_dict=True)["default"]

print("EMBEDDINGS TYPE: " + str(type(embeddings2)))

timer_sv = t.Timer()
print(f'Sess 2 start is {timer_sv}.')
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  search_vect = sess.run(embeddings2)
  print(f'Time elapsed is {timer_sv}.')
  try:
    search_vect = np.load('/Users/singhcpt/dev/semantic_analysis/search_vec.npy')
  except: 
    search_vect = sess.run(embeddings2)
    np.save('/Users/singhcpt/dev/semantic_analysis/search_vec.npy', search_vect)
print(f'Sess 2 end is {timer_sv}.')


cosine_similarities = pd.Series(cosine_similarity(search_vect, x).flatten())
output = ""
x = 0

print("\nRESULTS:\n")
for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
  x += 1
  print(i)
  print("RESULT " + str(x) + ": " + sentences[i] + "\n")
print(f'Total end is {timer_total}.')