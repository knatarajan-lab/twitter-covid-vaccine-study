# coding=utf-8

# Third-party imports
import numpy as np
from scipy.sparse import hstack
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# Project imports
from common_utils import load_covid_df
from resources.constants import LABEL_TARGET_MAP


def label2target(label):
    return LABEL_TARGET_MAP[label]


def train():
    ds = load_covid_df()

    count_vectorizer = CountVectorizer(ngram_range=(1, 4))
    vectorized_data = count_vectorizer.fit_transform(ds.text)
    indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))
    targets = ds.label.apply(label2target)
    data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.3,
                                                                          random_state=0)
    data_train_index = data_train[:, 0]
    data_train = data_train[:, 1:]
    data_train_index = data_test[:, 0]
    data_test = data_test[:, 1:]
    clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
    clf_output = clf.fit(data_train, targets_train)

    clf.score(data_test, targets_test)

    targets_pred = clf.predict(data_test)

    print("Accuracy:", metrics.accuracy_score(targets_test, targets_pred))
    print("Precision:", metrics.precision_score(targets_test, targets_pred, average='weighted'))
    print("Recall:", metrics.recall_score(targets_test, targets_pred, average='weighted'))


if __name__ == '__main__':
    train()
