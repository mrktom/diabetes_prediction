import pandas as pd
import numpy as np
import spacy
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
from cleaning import load_body_sentence
###

F_H5 = "prs_comp_tst.h5"
F_PKL = "test_feature.pkl" # for train use train_feature.pkl
T_PKL = "test_tfidf.pkl" #for train use train_tfidf.pkl


combined_dataframe = pd.read_hdf(F_H5)
###combined_dataframe = pd.read_hdf("prs_trn_2.h5")
print(combined_dataframe, combined_dataframe.info())
nlp = spacy.load('en_core_web_lg')
combined_dataframe['all_text'] = combined_dataframe['Headline'] +" "+ combined_dataframe['articleBody']
###y = combined_dataframe.values[:,5]
#countvectoriz+MultinomialNB for bi-classification
def cntv_MNB():
    vectorizer1 = CountVectorizer()
    vectorizer2 = CountVectorizer()
    countH = vectorizer1.fit_transform(combined_dataframe['Headline'])
    countB = vectorizer2.fit_transform(combined_dataframe['articleBody'])
    xheadline= countH.toarray()
    xbody=countB.toarray()
    x= np.hstack((xheadline,xbody))
    mnb=MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    y_predict=mnb.predict(X_test)
    print ('The accuracy of Naive Bayes Classifier is',mnb.score(X_test,y_test))
    #The accuracy of Naive Bayes Classifier is 0.7065898092798178
    print(classification_report(y_test,y_predict))
    '''
              precision    recall  f1-score   support

           0       0.86      0.72      0.78     10341
           1       0.46      0.66      0.54      3711

    accuracy                           0.71     14052
   macro avg       0.66      0.69      0.66     14052
weighted avg       0.75      0.71      0.72     14052
'''
#cntv_MNB()
#tfidf


##def tfidf(dataframe):
##    vocab_size=1000
##    vec = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=2,max_features=vocab_size)
##    vec.fit(dataframe["all_text"]) # Tf-idf calculated on the combined training + test set
##    vocabulary = vec.vocabulary_
##    vecH = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=2, vocabulary=vocabulary)
##    xHeadlineTfidf = vecH.fit_transform(dataframe['Headline']) # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
##    vecB = TfidfVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=2, vocabulary=vocabulary,max_features=vocab_size)
##    xBodyTfidf = vecB.fit_transform(dataframe['articleBody'])
##    xheadline= xHeadlineTfidf.toarray()
##    xbody=xBodyTfidf.toarray()
##    x= np.hstack((xheadline,xbody))
##    return x

###################################
#count words in a single string
def count_word(string):
    dic = defaultdict(int)
    for word in string.split():
        dic[word] += 1
    return dic
# idf-scaled overlap
# the idf-scaled overlap over the maximum possible idf-scaled overlap (will be between 1 and 0).
# raw count of overlapping words (no idf scaling)
# count of overlapping words divided by the maximum possible count of overlapping words (will be between 1 and 0)
def tfidf():
    vec = TfidfVectorizer(ngram_range=(1,1))
    tfidfmatrix = vec.fit_transform(combined_dataframe["articleBody"])
    word2tfidf = dict(zip(vec.get_feature_names(),vec.idf_))
    return word2tfidf
def overlap(title, body,word2tfidf):
    wordsH = count_word(title)
    wordsB = count_word(body)
    maximum, maximum_cnt = len(title), 0.0
    #calculate maximum possible scaled overlap by multiplying
    #the count of each word times its idf
    for (word, freq) in wordsH.items():
        if word in word2tfidf:
            temp = word2tfidf[word]
        else: 
            temp = 1
        maximum_cnt += freq * temp
    overlaps, overlap_cnt = 0, 0
    for (word, cnt_title) in wordsH.items():
        if word in wordsB:
            tf = min(cnt_title, wordsB[word])
            overlap_cnt += tf
            overlaps += tf * word2tfidf[word]
    features = [overlaps, overlaps / maximum, overlap_cnt, overlap_cnt / maximum_cnt]
    return features
##word2tfidf = tfidf()
##for index, data in combined_dataframe[0:5].iterrows():
##    feature1 = overlap(data['Headline'],data['articleBody'],word2tfidf)
##    feature += overlap(data['Headline'],data['articleBody'][:len(title) * 4],word2tfidf)
##    print(feature1)
##    print(len(feature1))

##############################################
##def word2vec(dataframe):
##    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
##    print("loaded successfully!")
##    docH = nlp.pipe(dataframe['Headline'])
##    print("pipe1 finish")
##    docB = nlp.pipe(dataframe['articleBody'])
##    print("pipe2 finish")
##    Headline_unigram = []
##    Body_unigram = []
##    print("loop1 going")
##    count = 0
##    for doc in docH:
##        tokens = [tokens.text for tokens in doc]
##        Headline_unigram.append(tokens)
##        count+=1
##        if count %1000==0:
##            print(count)
##    print("loop2 going")
##    count = 0
##    for doc in docB:
##        tokens = [tokens.text for tokens in doc]
##        Body_unigram.append(tokens)
##        count+=1
##        if count %1000==0:
##            print(count)
##    size = len(Headline_unigram)
##    headlineVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Headline_unigram))
##    headlineVec = np.array(headlineVec)
##    headlineVec = normalize(headlineVec)
##    bodyVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Body_unigram))
##    bodyVec = np.array(bodyVec)
##    bodyVec = normalize(bodyVec)
##    bdv = pd.DataFrame(bodyVec)
##    bdv.to_csv("body.csv",index = False)
##    hdv = pd.DataFrame(headlineVec)
##    hdv.to_csv("head.csv",index = False)
##    #x= np.hstack((xheadline,xbody) 
##    res = []
##    for i in range(0, size):
##        v1 = headlineVec[i].reshape(1, -1)
##        v2 = bodyVec[i].reshape(1,-1)
##        cs = cosine_similarity(v1,v2)
##        cs = cs[0][0]
##        res.append(cs)
##    resVec = np.array(res)
##    resv=pd.DataFrame(resVec)
##    resv.to_csv("csim.csv",index = False)
##    simVec2 = np.asarray(res)[:, np.newaxis]
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("word2vec loaded")
def sentence_similarity(title,sentence):
    sim = []
    for t in title.split():
        for s in sentence.split():
            if t in word2vec and s in word2vec:
                similarity = word2vec.similarity(t,s)
                sim.append(similarity)
    if len(sim) == 0:
        return False
    else:
        return sum(sim)/len(sim)
#sentence2vector("its a nice day", word2vec)
#file = "test_bodies.csv"
bs = load_body_sentence()
print("sentence of body generated")
def hb_similarities(title, body_sentences, word2vec,word2tfidf):
    max_overlap, max_overlap_cnt = 0, 0
    support = []
    for sub_body in body_sentences:
        similarity = sentence_similarity(title,sub_body)
        if similarity:
            support.append(similarity)
    if len(support) != 0:
        avg_cos = sum(support)/len(support)
    else:
        avg_cos = 0
    features = [max_overlap, max_overlap_cnt, max(support), min(support),avg_cos]
    return features
word2tfidf = tfidf()
print("word2tfidf generated")
def features():
    
    all_features = []
    ct =0
    for index, data in combined_dataframe.iterrows():
        title = data['Headline']
        feature = overlap(title,data['articleBody'],word2tfidf)
        feature += overlap(title,data['articleBody'][:len(title) * 4],word2tfidf)
        body_sentence = bs[data['Body ID']].split(',')
        feature += hb_similarities(title, body_sentence, word2vec,word2tfidf)
        all_features.append(feature)
        ct += 1
        if ct%500 ==0:
            print(ct)
    return all_features
ft = features()

output_feature = open(F_PKL, 'wb')
pickle.dump(ft, output)
output.close()
output_tfidf = open(T_PKL, 'wb')
pickle.dump(word2tfidf, output)
output.close()
