import fnmatch
import os
import pandas as pd
import regex as re
from nltk.corpus import stopwords


def fetch_reviews(testdata):
    path = 'op_spam_v1.4/'
    label = []

    # Fetch all the file paths of the .txt files and append in a list
    if testdata:
        file_paths = [os.path.join(subdir,f)
            for subdir, dirs, files in os.walk(path)
                for f in fnmatch.filter(files, '*.txt') if 'fold5' in subdir]
    else:
        file_paths = [os.path.join(subdir, f)
                      for subdir, dirs, files in os.walk(path)
                      for f in fnmatch.filter(files, '*.txt') if 'fold5' not in subdir]

    # Fetch all the labels and append in a list
    for path in file_paths:
        c = re.search('(trut|deceptiv)\w',path)
        label.append(c.group())

    # Create a dataframe of the label list
    labels = pd.DataFrame(label, columns=['Label'])

    # Fetch all the reviews and append in a list
    reviews = []
    for path in file_paths:
        with open(path) as f_input:
            reviews.append(f_input.read())

    # Create a dataframe of the review list
    reviews = pd.DataFrame(reviews, columns=['Review'])

    # Merge the review dataframe and label dataframe
    data = pd.merge(reviews, labels, right_index=True, left_index=True)
    # convert reviews to lowercase
    data['Review'] = data['Review'].map(lambda x: x.lower())
    # remove stopwords
    stop = stopwords.words('english')
    data['Review without stopwords'] = data['Review'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))

    return data


training_data = fetch_reviews(testdata=False)
print(training_data)
test_data = fetch_reviews(testdata=True)
print(test_data)