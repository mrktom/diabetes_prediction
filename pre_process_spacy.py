"""
COMP9417 19T1 Assignment 1 | Group Project: Fake News Challenge
Authors:
    Sudhan Maharjan - z5196539
    Qiqi Zhang -  z5185698
    Mark Thomas - z5194597
    Jianan Yan - z5168722

Description:
    Text preprocessing using Spacy Tokenizer
"""

import numpy
import os
import pandas
import platform
import re
import spacy
import string
from joblib import Parallel, delayed
from spacy.lang.en import stop_words
from spacy.tokens import Doc
from spacy.util import minibatch
from spacy_langdetect import LanguageDetector

###
# Change the values below to generate the pre-processed files
###
F_STANCES = "competition_test_stances.csv"
F_BODIES = "competition_test_bodies.csv"
O_H5 = "prs_comp_tst.h5"
O_CSV = "prs_comp_tst.csv"


# Adopted from: https://pypi.org/project/spacy-langdetect/ Custom Language Detector with spacy-langdetect
# def ggltrns_lang_detect(spacy_object):
#     assert isinstance(spacy_object, Doc) or isinstance(
#         spacy_object, Span), "spacy_object must be a spacy Doc or Span object but it is a {}".format(type(spacy_object))
#     detection = Translator().detect(spacy_object.text)
#     return {'language':detection.lang, 'score':detection.confidence}

class SpacyTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        if not any(ppnm == "lang_detector" for ppnm in dflt_nlp.pipe_names):
            dflt_nlp.add_pipe(LanguageDetector(), name="lang_detector", last=True)

        dflt_tok = dflt_nlp.Defaults.create_tokenizer(dflt_nlp)
        lng_ck = dflt_nlp(text)
        if lng_ck._.language['language'] == r"en":
            text = """""".join(
                ele if ele not in ['\\', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '—'] and ele.isdigit() == False
                       and ele not in [pct for pct in string.punctuation if pct not in ["'", "—", '–', '-']] else ' '
                for ele in text)
            aft_tk = """ """.join(re.sub(r'[\W+\d]', '', k.lower_) for k in dflt_tok(text)
                                  if
                                  re.sub(r'[\W+\d]', '', k.lower_) != '' and re.sub(r'[\W+\d]', '', k.lower_) not in (
                                      stop_words.STOP_WORDS))
            words = [tk.lemma_ for tk in dflt_tok(aft_tk) if len(tk.lemma_) > 2]
            return Doc(self.vocab, words=words)
        else:
            return Doc(self.vocab, words='')


fl_sep = os.path.dirname(os.path.abspath(__file__)) + ('\\' if platform.system() == 'Windows' else '/')
art_bod = pandas.read_csv(f"{fl_sep}competition_test_bodies.csv")
art_bod.sort_values('Body ID', inplace=True)
art_bod = art_bod[['Body ID'] + art_bod.drop('Body ID', 1).columns.tolist()]
art_bod = art_bod.reset_index(drop=True)
print(f"This is art_bod: \n{art_bod}\n\n")
print(f"This is its info: \n{art_bod.info()}\n\n")

art_head = pandas.read_csv(f"{fl_sep}competition_test_stances.csv")
art_head.sort_values('Body ID', inplace=True)
art_head = art_head[['Body ID'] + art_head.drop('Body ID', 1).columns.tolist()]
art_head = art_head.reset_index(drop=True)
print(f"This is art_head: \n{art_head}\n\n")
print(f"This is its info: \n{art_head.info()}\n\n")


# def FinalData(coldata, model):
#     return model[0].pipe(coldata)

def FinalData(coldata):
    return [tk for tk in nlp.pipe(coldata)]


if __name__ == "__main__":
    global nlp, dflt_nlp

    dflt_nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "ner", "textcat"])
    prs_bod = art_bod['articleBody'].tolist()

    cnt = 0
    nlp.tokenizer = SpacyTokenizer(nlp.vocab)

    prt_bod = minibatch(prs_bod, size=1)
    execute = Parallel(n_jobs=-1, backend="threading", prefer="threads", verbose=art_head.shape[0])
    do = delayed(FinalData)
    tasks = (do(vl) for _, vl in enumerate(prt_bod))
    res_bod = execute(tasks)
    # print(f"Final: {[tk.text for doc in t[0] for tk in doc]}\n Final_len: {len(t)}\n")
    ept_rws = [r for r in range(len(res_bod)) if not len(res_bod[r][0])]
    del_bid = [art_bod['Body ID'][rw] for rw in ept_rws]
    fnl_t = [doc for r in range(len(res_bod)) for doc in res_bod[r] if len(res_bod[r][0])]
    print(f"The body rows to be dropped: {ept_rws}\nTheir body_ids are: {del_bid}\n")
    print(f"The fnl_t is: {fnl_t}\n Fnl_t len: {len(fnl_t)}\n")
    art_bod.drop(index=ept_rws, inplace=True)
    art_bod = art_bod.reset_index(drop=True)
    print(f"Updated art_bod: \n{art_bod}\n\n")
    art_bod['articleBody'] = numpy.array([" ".join(tk.text for tk in rw) for rw in fnl_t])
    print(f"Final art_bod: \n{art_bod}\n\n")

    art_head = art_head[~art_head['Body ID'].isin(del_bid)]
    art_head = art_head.reset_index(drop=True)
    print(f"After deleted art_head: \n{art_head}\n\n")
    prs_head = art_head['Headline'].tolist()
    prt_head = minibatch(prs_head, size=1)
    tasks = (do(vl) for _, vl in enumerate(prt_head))
    res_head = execute(tasks)
    ept_rws = [r for r in range(len(res_head)) if not len(res_head[r][0])]
    fnl_t = [doc for r in range(len(res_head)) for doc in res_head[r] if len(res_head[r][0])]
    print(f"The headline rows to be dropped: {ept_rws}\n They have length: {len(ept_rws)}\n")
    print(f"The fnl_t is: {fnl_t}\n Fnl_t len: {len(fnl_t)}\n")
    art_head.drop(index=ept_rws, inplace=True)
    art_head = art_head.reset_index(drop=True)
    art_head["Headline"] = numpy.array([" ".join(tk.text for tk in rw) for rw in fnl_t])
    print(f"Final art_head: \n{art_head}\n\n")
    unk_bdid = art_head['Body ID'].unique().tolist()
    print(f"The unqiue body id is: {unk_bdid}\n")
    art_bod = art_bod[art_bod['Body ID'].isin(unk_bdid)]  # In case there are now no headlines for a unique body id
    art_bod = art_bod.reset_index(drop=True)
    print(f"Updated final art_bod: \n{art_bod}\n\n")
    com_df = pandas.merge(art_bod, art_head, on='Body ID', how='inner')
    print(f"Combined dataframe: \n{com_df}\n\n", f"This is its info: \n{com_df.info()}\n\n")
    # com_df.to_pickle(f"{fl_sep}prs_trn_1.pkl")
    com_df.to_hdf(f"{fl_sep}prs_comp_tst.h5", key="com_df", format="fixed", complib="zlib", complevel=9)
    com_df.to_csv(f"{fl_sep}prs_comp_txt.csv")
    # rd_pk = pandas.read_pickle(f"{fl_sep}prs_trn.pkl")
    # print(f"This is rd_pk: \n{rd_pk}\n\nThis is its info: {rd_pk.info()}\n\n")
