import numpy as np
import re
import string
from pathlib import Path
import pandas as pd
from pandas.core.common import random_state
from sklearn.feature_extraction.text import CountVectorizer

# from liwc import Liwc
# liwc = Liwc("./liwc.dic")


# Important module for preprocessing
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import nltk.data
# nltk.download('punkt')
# sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def read_into_pandas_dataframe(data=None):
    # Give path yourself, or find the data in the folder from github
    #
    # expects that the data is in a folder like this
    #
    # ./data/
    # ./data/truthful
    # ./data/deceptive
    #
    # otherwise it won't work
    if data is None:
        data_path = Path.cwd() / "data"
    else:
        data_path = Path(data)

    data_dict = {}
    hidden_file = "\."
    for folds in data_path.iterdir():
        match = re.match(hidden_file, folds.name)
        if not match:
            data_dict[folds.name] = {}
            # print(data_dict)
            # print(folds)
            for fold in folds.iterdir():
                match = re.match(hidden_file, fold.name)
                if not match:
                    # print(fold.name)
                    data_dict[folds.name][fold.name] = []
                    # print(data_dict)
                    for review in fold.glob("*"):
                        # print(review)
                        with open(review, "r") as rev:
                            rev_string = rev.read()
                            # print(string)
                            data_dict[folds.name][fold.name].append(rev_string)
                    # print(fold.name)
                    # print(len(data_dict[folds.name][fold.name]))

    df_truthful = pd.DataFrame.from_dict(data_dict['truthful'])
    df_truthful['label'] = [1] * len(df_truthful)

    df_deceptive = pd.DataFrame.from_dict(data_dict['deceptive'])
    df_deceptive['label'] = [0] * len(df_truthful)

    df_stacked = pd.concat([df_truthful,df_deceptive])
    df_stacked = df_stacked.reindex(sorted(df_stacked.columns), axis=1)
    df_stacked = df_stacked.reset_index(drop=True)
    return df_stacked


def preprocessing(df,
                  strip=True,
                  lower_case=False,
                  del_stopwords=False,
                  del_punkt=False,
                  del_numbers=False,
                  stemming=False,
                  pos_tagging=False,
                  liwc_cats=False,
                  ngrams=1):
    """
    Should return at least a document term dataframe per fold, including columns
    for parts of speech tags and liwc categories if selected.
    """

    # strips whitespace from reviews
    if strip:
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: review.strip())

    # makes all strings lowercase
    if lower_case:
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: review.lower())

    # remove stopwords
    if del_stopwords:
        # print(stopwords.words('english'))
        sw_list = stopwords.words('english')
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: ' '.join([word for word in word_tokenize(review) if word not in sw_list]))

    # removes punctuation marks from review strings
    if del_punkt:
        regex = re.compile('\w*[%s]+(\w*\s)' % re.escape(string.punctuation))
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: regex.sub('', review))

    # removes numbers from string
    if del_numbers:
        for fold in df.columns[:-1]:
            # deletes numbers surrounded by spaces
            df[fold] = df[fold].apply(lambda review: re.sub("\s\d+\s", ' ', review))
            # deletes words containing a number
            df[fold] = df[fold].apply(lambda review: re.sub("\S*\d+\S*", '', review))

    # stems words to reduce features later
    if stemming:
        stemmer = PorterStemmer()
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: ' '.join([stemmer.stem(word) for word in word_tokenize(review)]))

    # pos betekent part of speech, maakt eigenlijk gewoon extra features om van te leren
    if pos_tagging:
        for fold in df.columns[:-1]:
            df[fold+'_pos'] = df[fold].apply(lambda review: nltk.pos_tag(word_tokenize(review)))
            # print(df.head())
        df = df.reindex(sorted(df_stacked.columns), axis=1)

    # First we construct the corpus document term matrix, and then split it into fold matrices again
    if pos_tagging:
        df_corpus = pd.DataFrame.from_dict({'reviews': [], 'pos_tags': [], 'label': []})
        step = 2
    else:
        df_corpus = pd.DataFrame.from_dict({'reviews': [], 'label': []})
        step = 1


    for fold in df.columns[:-1:step]:
        if pos_tagging:
            df_fold = df[[fold, fold+'_pos', 'label']]
            df_fold = df_fold.rename(columns={fold:'reviews', fold+'_pos':'pos_tags'})
        else:
            print(fold)
            df_fold = df[[fold, 'label']]
            df_fold = df_fold.rename(columns={fold:'reviews'})
        df_corpus = pd.concat([df_corpus, df_fold], ignore_index=True)

    print(df_corpus.shape)
    print(df_corpus.head())

    vectorizer = CountVectorizer()
    reviews = df_corpus['reviews']
    review_count_vector = vectorizer.fit_transform(reviews)
    corpus_tm = pd.DataFrame(review_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
    corpus_tm = corpus_tm.transpose()
    # print(corpus_tm.shape)
    # print(corpus_tm.head())

    if pos_tagging:
        pos_tags = df_corpus['pos_tags'].apply(lambda review: ' '.join(['pos_' + tagged_word[1] for tagged_word in review]))
        pos_tags_count_vector = vectorizer.fit_transform(pos_tags)
        df_pos_tags = pd.DataFrame(pos_tags_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
        df_pos_tags = df_pos_tags.transpose()
        corpus_tm = pd.concat([corpus_tm, df_pos_tags], axis=1)

    if ngrams > 1:
        ngram_reviews = df_corpus['reviews']
        for ngram_size in range(2, ngrams + 1):
            vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size))
            ngram_reviews_count_vector = vectorizer.fit_transform(ngram_reviews)
            df_ngram_reviews = pd.DataFrame(ngram_reviews_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
            df_ngram_reviews = df_ngram_reviews.transpose()
            corpus_tm = pd.concat([corpus_tm, df_ngram_reviews], axis=1)
            # print(df_ngram_reviews.shape)
            # print(df_ngram_reviews.head())
            # for ngram in df_ngram_reviews.iloc[0][df_ngram_reviews.iloc[0] > 0].index:
            #     print(ngram)




    # First 4 * 160 documents are training data
    X_train_corpus = corpus_tm[:640].to_numpy()
    y_train_corpus = np.array(df_corpus['label'][:640])

    X_test_corpus = corpus_tm[640:].to_numpy()
    y_test_corpus = np.array(df_corpus['label'][640:])

    # List of term matrices of first 4 folds, and their labels
    X_dev_folds = []
    y_dev_folds = []
    for i in range(0, len(corpus_tm) - 160, 160):

        X_dev_fold = corpus_tm[i:i+160]
        X_dev_folds.append(X_dev_fold.to_numpy())

        y_dev_fold = df_corpus['label'][i:i+160]
        y_dev_folds.append(y_dev_fold)

    return df_corpus, corpus_tm, X_train_corpus, y_train_corpus, X_test_corpus, y_test_corpus, X_dev_folds, y_dev_folds

df_stacked = read_into_pandas_dataframe()

df_corpus, corpus_tm, X_train_corpus, y_train_corpus, X_test_corpus, y_test_corpus, X_dev_folds, y_dev_folds = preprocessing(df_stacked,
                del_punkt=True,
                lower_case=True,
                del_numbers=True,
                del_stopwords=True,
                stemming=False,
                pos_tagging=False,
                ngrams=1)


def modelling_experiment(X_train, y_train, X_test, y_test):
    print(X_train.shape)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(accuracy_score(y_test, y_pred))

modelling_experiment(X_train_corpus, y_train_corpus, X_test_corpus, y_test_corpus)
