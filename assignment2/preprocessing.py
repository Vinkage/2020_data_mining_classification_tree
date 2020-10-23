import re
import string
from pathlib import Path
import pandas as pd
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

    # Here we assemble the document term count dataframe, which we will use to let our models learn
    total_tm = []
    folds_tm = []
    labels = df[df.columns[-1]]

    # print(labels)
    # print(df.columns)
    vectorizer = CountVectorizer()
    for fold in df.columns[:-1:2]:
        reviews = df[fold]
        review_count_vector = vectorizer.fit_transform(reviews)
        df_fold = pd.DataFrame(review_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
        df_fold = df_fold.transpose()
        if pos_tagging:
            pos_tags = df[fold+'_pos'].apply(lambda review: ' '.join(['pos_' + tagged_word[1] for tagged_word in review]))
            pos_tags_count_vector = vectorizer.fit_transform(pos_tags)
            df_pos_tags = pd.DataFrame(pos_tags_count_vector.toarray().transpose(), index=vectorizer.get_feature_names())
            df_pos_tags = df_pos_tags.transpose()
            df_fold = pd.concat([df_fold, df_pos_tags], axis=1)
        folds_tm.append(df_fold)

        # print(df_fold.shape)
        # print(df_fold.head())
    return folds_tm, labels

df_stacked = read_into_pandas_dataframe()
tm_of_folds, labels = preprocessing(df_stacked,
                                    del_punkt=True,
                                    lower_case=True,
                                    del_numbers=True,
                                    del_stopwords=True,
                                    stemming=False,
                                    pos_tagging=True,
                                    ngrams=1)


def modelling_experiment(tm_of_folds, labels):
    # print(type(tm_of_folds[0].to_))
    tree = DecisionTreeClassifier()
    tree.fit(tm_of_folds[0].to_numpy(), np.array(labels))
    # y_true = labels.to_numpy()
    # print(y_true)

    # y_pred = tree.predict(tm_of_folds[-1], labels)
    # print(accuracy_score())

modelling_experiment(tm_of_folds, labels)
