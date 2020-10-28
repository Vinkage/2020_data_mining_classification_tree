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
    # print(df_stacked.head())
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

    # pos betekent part of speech, maakt eigenlijk gewoon extra features om van te leren
    if pos_tagging:
        pos_tag_df = pd.DataFrame()
        for fold in df.columns[:-1]:
            pos_tag_df[fold+'_pos'] = df[fold].apply(lambda review: nltk.pos_tag(word_tokenize(review)))
            # print(df.head())

    # remove stopwords
    if del_stopwords:
        # print(stopwords.words('english'))
        sw_list = stopwords.words('english')
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: ' '.join([word for word in word_tokenize(review) if word not in sw_list]))

    # removes punctuation marks from review strings
    if del_punkt:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: ' '.join([regex.sub('', word) for word in word_tokenize(review)]))

    # removes numbers from string
    if del_numbers:
        for fold in df.columns[:-1]:
            # deletes numbers surrounded by spaces
            df[fold] = df[fold].apply(lambda review: re.sub("\s\d+\s", ' ', review))
            # deletes words containing a number
            # df[fold] = df[fold].apply(lambda review: re.sub("\S*\d+\S*", '', review))

    # stems words to reduce features later
    if stemming:
        stemmer = PorterStemmer()
        for fold in df.columns[:-1]:
            df[fold] = df[fold].apply(lambda review: ' '.join([stemmer.stem(word) for word in word_tokenize(review)]))


    # First we construct the corpus document term matrix, and then split it into fold matrices again
    df_corpus = pd.DataFrame.from_dict({'reviews': [], 'label': []})


    # Here we construct the corpus term matrix from the folds
    for fold in df.columns[:-1]:
        if pos_tagging:
            df_fold = pd.concat([df[[fold]], pos_tag_df[[fold+'_pos']]], axis=1)
            df_fold = pd.concat([df_fold, df[['label']]], axis=1)
            # print("the fold df now has the columns:", df_fold.columns)
            # df_fold = df[[fold, fold+'_pos', 'label']]
            df_fold = df_fold.rename(columns={fold:'reviews', fold+'_pos':'pos_tags'})
        else:
            # print(fold)
            df_fold = df[[fold, 'label']]
            df_fold = df_fold.rename(columns={fold:'reviews'})
        df_corpus = pd.concat([df_corpus, df_fold], ignore_index=True)

    # Here we count the frequency of the terms in the processed reviews
    vectorizer = CountVectorizer()
    reviews = df_corpus['reviews']
    review_count_vector = vectorizer.fit_transform(reviews)
    corpus_tm = pd.DataFrame(review_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
    corpus_tm = corpus_tm.transpose()

    # Here we add the frequency of the pos tags to the corpus term matrix
    if pos_tagging:
        pos_tags = df_corpus['pos_tags'].apply(lambda review: ' '.join([tagged_word[1] for tagged_word in review if tagged_word[0] not in string.punctuation]))
        pos_tags_count_vector = vectorizer.fit_transform(pos_tags)
        df_pos_tags = pd.DataFrame(pos_tags_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
        df_pos_tags = df_pos_tags.transpose()
        # df_pos_tags = df_pos_tags.drop(['pos_'], axis=1)
        corpus_tm = pd.concat([corpus_tm, df_pos_tags], axis=1)

    # Here we handle ngrams
    if ngrams > 1:
        ngram_reviews = df_corpus['reviews']
        for ngram_size in range(2, ngrams + 1):
            vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size))
            ngram_reviews_count_vector = vectorizer.fit_transform(ngram_reviews)
            df_ngram_reviews = pd.DataFrame(ngram_reviews_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
            df_ngram_reviews = df_ngram_reviews.transpose()
            corpus_tm = pd.concat([corpus_tm, df_ngram_reviews], axis=1)

    # If you choose ngrams=0, then only the pos tags are in the term matrix
    elif ngrams == 0:
        pos_tags = df_corpus['pos_tags'].apply(lambda review: ' '.join([tagged_word[1] for tagged_word in review if tagged_word[0] not in string.punctuation]))
        pos_tags_count_vector = vectorizer.fit_transform(pos_tags)
        df_pos_tags = pd.DataFrame(pos_tags_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
        df_pos_tags = df_pos_tags.transpose()
        corpus_tm = df_pos_tags
        print(corpus_tm.columns)


    # First 4 * 160 documents are training data
    X_train = corpus_tm[:640].to_numpy()
    y_train = np.array(df_corpus['label'][:640])

    X_test = corpus_tm[640:].to_numpy()
    y_test = np.array(df_corpus['label'][640:])

    return df_corpus, corpus_tm, X_train, y_train, X_test, y_test

if __name__=='__main__':
    df_stacked = read_into_pandas_dataframe()

    df_corpus, corpus_tm, X_train, y_train, X_test, y_test = preprocessing(df_stacked,
                    lower_case=True,
                    pos_tagging=False,
                    del_stopwords=True,
                    del_punkt=True,
                    del_numbers=True,
                    stemming=False,
                                                                           ngrams=2)
    print(corpus_tm.shape)
