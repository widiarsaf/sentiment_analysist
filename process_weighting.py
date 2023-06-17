from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def splitDataset(sentiment, labels):
    train_X, test_X, train_Y, test_Y = train_test_split(
        sentiment, labels, test_size=0.2, random_state=6777)
    df_train = pd.DataFrame()
    df_train['sentiment'] = train_X
    df_train['labels'] = train_Y
    df_test = pd.DataFrame()
    df_test['sentiment'] = test_X
    df_test['labels'] = test_Y
    train_Y = np.nan_to_num(train_Y)
    return [df_train, df_test, train_X, test_X, train_Y, test_Y]


def tf_idf(df_train, df_test, sentiment):
    tfidf_vect_data = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8',
                                      lowercase=True, max_df=1.0, max_features=30000, min_df=5,
                                      ngram_range=(1, 1), norm='l2',
                                      strip_accents='unicode', sublinear_tf=False,
                                      token_pattern='\\w{1,}', tokenizer=None, use_idf=True,
                                      vocabulary=None)
    tfidf_vect_data = tfidf_vect_data.fit(sentiment['sentiment'])
    train_X_tfidf_data = tfidf_vect_data.transform(df_train['sentiment'])
    test_X_tfidf_data = tfidf_vect_data.transform(df_test['sentiment'])
    test_x_arr = test_X_tfidf_data.toarray()
    train_x_arr = train_X_tfidf_data.toarray()

    train_shape = train_X_tfidf_data.shape
    test_shape = test_X_tfidf_data.shape
    return test_x_arr, train_x_arr, train_shape, test_shape
