# Installing all necessary libraries and loading ACTER

import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string
import spacy
! git clone https://github.com/AylaRT/ACTER.git
import os
import pandas as pd


domains = ["corp", "equi", "wind","htfl"]
langs   = ["en", "fr", "nl"]
domain   = domains[0]       # Selecting a domain
language = langs[0]
texts=[]
file_names=[]

folder_path = "/content/ACTER/" + language + "/" + domain + "/annotated/texts"   #unannotated_texts       annotated/texts_tokenised

file_list = os.listdir(folder_path)

for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            texts.append(text.replace("  ", " ").replace(" -","-").replace(" - ","-"))
import csv
true_terms=[]
ann_path = '/content/ACTER/' + language + "/" + domain + "/annotated/annotations/unique_annotation_lists/" + domain + "_"+language+"_terms_nes.tsv"
with open(ann_path, 'r', newline='') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        true_terms.append(row[0].lower())

true_terms_l = set([w.lower().replace("  "," ").replace("- ","-") for w in true_terms])
true_terms_mwe = set ([w for w in true_terms_l if (len(w.split(" "))>1)]+ [w for w in true_terms_l if len(w.split("-"))>1 ])
true_terms_uni = set([w for w in true_terms_l if w not in true_terms_mwe ])

print('True terms all: ', len(true_terms_l))
print('True terms uni: ', len(true_terms_uni))
print('True terms mwe: ', len(true_terms_mwe))

punc = list(string.punctuation)

#####################################################################################################################################




#____________________________________Loading stop words depending on language____________________________________________________

import requests

if language=="en":
      url = 'https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt'
      stop_words = (requests.get(url).text).split(",")
      nlp = spacy.load("en_core_web_sm")

elif language=="fr":
      !python3 -m spacy download fr_core_news_sm
      from spacy.lang.fr.examples import sentences
      url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
      stop_words = (requests.get(url).text).split("\n")
      nlp = spacy.load("fr_core_news_sm")

elif language=="nl":
      !python3 -m spacy download nl_core_news_sm
      from spacy.lang.nl.examples import sentences
      url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-nl/master/stopwords-nl.txt'
      stop_words = (requests.get(url).text).split("\n")
      nlp = spacy.load("nl_core_news_sm")
#####################################################################################################################################



#____________________________________Unigram Extraction____________________________________________________

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter

# Modify tokenizer infix patterns
infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # Commented out regex that splits on hyphens between letters:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

import regex as re

pattern = r'^[\p{L}\d]+$'

def filter_words(words):
    filtered = []
    for word in words:
        digits = sum(c.isdigit() for c in word)
        if len(word)-digits >= digits:
            filtered.append(word)
    return filtered


unigrams=[]
abb=[]
for tex in texts:
        text_token=nlp(tex)
        text_token=[i.text.lower() for i in text_token if i.pos_ in ["ADJ","ADV","NOUN","PROPN","VERB","X","INTJ"]]
        text_token=[w for w in text_token if w not in stop_words and
                                             len(set(w).intersection(set(punc)))==0 and
                                             re.match(pattern, w)]
        text_token= filter_words(text_token)
        unigrams+=text_token

print(len(set(unigrams)))




#____________________________________Semantic filtering based on stop words____________________________________________________

if language=="en":
      !pip install gensim
      from gensim.models import KeyedVectors
      import gensim.downloader as api
      fasttext  = api.load("fasttext-wiki-news-subwords-300")

elif language=="fr":
     !pip install fasttext
     import fasttext
     from huggingface_hub import hf_hub_download
     model_path = hf_hub_download(repo_id="facebook/fasttext-fr-vectors", filename="model.bin")
     fasttext = fasttext.load_model(model_path)

elif language=="nl":
     !pip install fasttext
     import fasttext
     from huggingface_hub import hf_hub_download
     model_path = hf_hub_download(repo_id="facebook/fasttext-nl-vectors", filename="model.bin")
     fasttext = fasttext.load_model(model_path)


import numpy as np

words1 =[i for i in list(set(unigrams)) if i in fasttext ]
words2 = [i for i in stop_words if i in fasttext ]

def get_embeddings(words):
    embeddings = []
    for word in words:
        if word in fasttext:
            embeddings.append(fasttext[word])
        else:
            embeddings.append(None)
    return embeddings

def cosine_similarity_matrix(embeddings1, embeddings2):
    embeddings1 = [e for e in embeddings1 if e is not None]
    embeddings2 = [e for e in embeddings2 if e is not None]

    embeddings1_array = np.array(embeddings1)
    embeddings2_array = np.array(embeddings2)

    dot_product_matrix = np.dot(embeddings1_array, embeddings2_array.T)

    magnitudes1 = np.linalg.norm(embeddings1_array, axis=1)
    magnitudes2 = np.linalg.norm(embeddings2_array, axis=1)

    magnitudes_matrix = np.outer(magnitudes1, magnitudes2)

    cosine_sim_matrix = dot_product_matrix / magnitudes_matrix

    return cosine_sim_matrix

def find_similar_words(words1, words2, threshold):
    embeddings1 = get_embeddings(words1)
    embeddings2 = get_embeddings(words2)

    similarity_matrix = cosine_similarity_matrix(embeddings1, embeddings2)

    similar_words = []
    for i in range(len(words1)):
        for j in range(len(words2)):
            if embeddings1[i] is not None and embeddings2[j] is not None and similarity_matrix[i, j] > threshold:
                similar_words.append(words1[i])
                break

    return similar_words

th=0.5
similar_words = find_similar_words(words1, words2, threshold=th)
print(f'Unigrams with cosine similarity higher than {th} with any stop-word:')
print(similar_words)

#####################################################################################################################################



#____________________________________Evaluation____________________________________________________

def calculate_metrics(true_terms, extracted_terms):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score


uni = set(unigrams)-set(similar_words)

precision, recall, f1_score=calculate_metrics(true_terms_uni, uni)
print(sorted(set(uni)))
print(len(set(uni)))
print("Precision:", round(precision*100,2))
print("Recall:", round(recall*100,2))
print("F1 Score:", round(f1_score*100,2))

#####################################################################################################################################



#____________________________________Visualization of results____________________________________________________

ttu_specific = []          # Specific_Term
ttu_comon = []             # Common_Term
ttu_NE = []                # Named_Entity
ttu_ood = []               # OOD_Term
ttu = []

true_terms=[]
ann_path = '/content/ACTER/' + language + "/" + domain + "/annotated/annotations/unique_annotation_lists/" + domain + "_"+language+"_terms_nes.tsv"
with open(ann_path, 'r', newline='') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        if len(row[0].split(" "))==1 and len(row[0].split("-"))==1 and len(row[0].split("'"))==1:
            if row[1]=="Specific_Term":
                ttu_specific.append(row[0].lower())

            if row[1]=="Common_Term":
                ttu_comon.append(row[0].lower())

            if row[1]=="Named_Entity":
                ttu_NE.append(row[0].lower())

            if row[1]=="OOD_Term":
                ttu_ood.append(row[0].lower())
            ttu.append(row[0].lower())

ttu_specific = set(ttu_specific)         # Specific_Term
ttu_comon = set(ttu_comon)             # Common_Term
ttu_NE = set(ttu_NE)                # Named_Entity
ttu_ood = set(ttu_ood)             # OOD_Term
ttu = set(ttu)

import matplotlib.pyplot as plt

x = []
p = []
r = []
f1 = []

false = []
sp = []
common = []
ne = []
ood = []

for i in range(0,105,5):
      th=i/100
      similar_words = find_similar_words(words1, words2, threshold=th)

      uni= set(unigrams)-set(similar_words)

      precision, recall, f1_score=calculate_metrics(true_terms_uni, uni)

      x.append(th)
      p.append(precision)
      r.append(recall)
      f1.append(f1_score)

      false.append( len(uni-true_terms_l)/len(uni))
      sp.append( len(uni.intersection(ttu_specific))/len(uni))
      common.append( len(uni.intersection(ttu_comon))/len(uni))
      ne.append( len(uni.intersection(ttu_NE))/len(uni))
      ood.append( len(uni.intersection(ttu_ood))/len(uni))
      if len(uni-true_terms_l)+len(uni.intersection(ttu_specific))+len(uni.intersection(ttu_comon))+len(uni.intersection(ttu_NE))+len(uni.intersection(ttu_ood))  <len(uni):
        print(uni - (uni-true_terms_l)-ttu_specific-ttu_comon-ttu_NE-ttu_ood)


#____________________________________Figure 1_______________________________________________________

plt.figure(figsize=(10, 6))
plt.axvline(x=0.4, color='purple', linewidth=1)
plt.plot(x, p, label='Precision', marker='o')
plt.plot(x, r, label='Recall', marker='s')
plt.plot(x, f1, label='F1 score', marker='^')

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Metrics', fontsize=12)
plt.title(f'{domain}({language})', fontsize=14)

plt.legend(fontsize=10)
plt.tight_layout()

plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7)
plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.05, 1))

ax = plt.gca()
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.spines['top'].set_color('lightgrey')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_linewidth(0.7)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.show()


#____________________________________Figure 2_______________________________________________________

import matplotlib.pyplot as plt
import numpy as np

x = np.array(x)
false = np.array(false)
sp = np.array(sp)
common = np.array(common)
ne = np.array(ne)
ood = np.array(ood)

total = false + sp + common + ne + ood
false /= total
sp /= total
common /= total
ne /= total
ood /= total

plt.figure(figsize=(10, 6))
plt.stackplot(x, ne, common, sp, ood, false,
              labels=['NE', 'Common', 'Specific', 'OOD', 'False Positive'],
              alpha=0.7,
              colors=['#d62728', '#2ca02c', '#1f77b4', 'black', '#d9d9d9'])

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.axvline(x=0.4, color='purple', linewidth=1)

plt.title(f'{domain}({language})', fontsize=14)
plt.legend(loc='upper left', fontsize=10)

plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.05, 1))

ax = plt.gca()
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.spines['top'].set_color('lightgrey')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_linewidth(0)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

plt.show()



























