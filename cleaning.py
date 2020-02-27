import pandas as pd
import spacy
import re
import string
from spacy.lang.en import stop_words
F_BODIES = 'competition_test_bodies.csv'
def load_body_sentence():
    nlp = spacy.load('en_core_web_lg')
    df=pd.read_csv(F_BODIES)
    #df=pd.read_csv('test_bodies.csv')
    body_sentence = {}
    for row in df.iterrows():
        if row[1]["Body ID"] not in body_sentence:
            line = row[1]["articleBody"]
            #print(f"Line: {line}\n")
            text = """""".join(ele if ele not in ['\\' , '\a', '\b', '\f', '\r' '\t', '\v'] \
                               and not ele.isdigit() else ' ' for ele in line)
            #print(f"Pre-processed text: {text}\n")
            nlpDoc = nlp(text)
            temp = [(" ".join([re.sub(r'[\W+\d]','', token.text) for token in sent if \
                               re.sub(r'[\W+\d]','', token.text) != '' and \
                               re.sub(r'[\W+\d]','', token.text) not in (stop_words.STOP_WORDS) and \
                               len(re.sub(r'[\W+\d]', '', token.text)) > 2])).lower()
                               for sent in nlpDoc.sents]
            #print(f"Temp tokens: {temp}\n")
            nw_temp = [" ".join(tk.lemma_ for tk in sent) for item in temp for sent in [nlp(item)]]
            temp = ','.join([x for x in temp if x != ''])
            #print(f"Final: {temp}\n")
            body_sentence[row[1]["Body ID"]] = temp
    return body_sentence




