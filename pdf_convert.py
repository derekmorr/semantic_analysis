import os
import logging
import pdf_converter
import pickle
import re
import spacy
from spacy.lang.en import English
import tensorflow_hub as hub
import Timer as t

def convert(file_path):
    try:
        with open("sentences.txt", 'rb') as f:
            sentences = pickle.load(f)
    except:
        pdf_texts = [] 
        sentences = []
        timer_convert = t.Timer()
        print(f'PDF conversion start: {timer_convert}.')
        for filename in os.listdir(file_path):
            if (".pdf" in filename):
                print("Converting " + filename + " to pdf.") 
                print(file_path + "/" + filename)
                pdf_texts.append(pdf_converter.convert(file_path + "/" + filename))
            if (".txt" in filename):
                print("Converting " + filename + " to array.")
                print(file_path + "/" + filename)
                with open(file_path + "/" + filename) as f:
                    for line in f:
                        sentences.append(line)
                return sentences
        print(f'PDF conversion end: {timer_convert}.')

        timer_spacy = t.Timer()
        print(f'Spacy load start: {timer_spacy}.')
        # python -m spacy download en_core_web_md you will need to install this on first load
        nlp = spacy.load('en_core_web_md')
        logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow
        timer_convert = t.Timer()
        print(f'Spacy load end: {timer_spacy}.')

        timer_nlp = t.Timer()
        text = ""
        for pdf_text in pdf_texts:
            text += pdf_text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
            text += "\n"
        text = ' '.join(text.split())
        nlp.max_length = 1000000000
        doc = nlp(text)

        for i in doc.sents:
            if len(i) > 1:
                sentences.append(i.string.strip())

        with open("sentences.txt", 'wb') as f:
            pickle.dump(sentences, f)

    return sentences
