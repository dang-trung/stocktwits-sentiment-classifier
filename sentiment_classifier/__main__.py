#!/usr/bin/env python3
"""Random Forest Sentiment Classifier.

This module downloads +2m StockTwits messages related to
cryptocurrencies and classifies their sentimentes using Random Forest.
Enters in cmd line: python -m sentiment_classfiier"""
import warnings

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from .data import get_stocktwits_message
from .process import pre_process

if __name__ == '__main__':
    symbols = pd.read_csv('data/00_external/symbols.csv', header=None)[1]
    symbols = symbols[symbols.str.endswith('.X')]  # find crypto symbols
    symbols = symbols.to_list()  #
    start = "2014-11-28"
    end = "2020-07-26"
    for symbol in symbols:
        try:
            get_stocktwits_message(symbol=symbol, start=start, end=end,
                                   file_name=f"data/01_raw/stocktwits/"
                                             f"{symbol[:-2]}.csv")
        except IndexError:
            print(f"No message about {symbol} on StockTwits.")
            print("------------")
    print(
        f"Combining files containing messages about single symbols to a "
        f"master file...")
    print("------------")
    combined_csv = pd.concat(
        [pd.read_csv(f"data/01_raw/stocktwits/{symbol[:-2]}.csv") for symbol in
         symbols], ignore_index=True)
    combined_csv.to_csv("data/01_raw/stocktwits.csv")

    print("Start classifying sentiment...")
    print("------------")
    print("Importing StockTwits messages...")
    print("------------")
    # Avoid FutureWarning (future conflicts between numpy and pandas)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Avoid False SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    # Import data
    stocktwits = pd.read_csv("data/01_raw/stocktwits.csv", index_col=0)
    # Extract all messages tagged with sentiment by user
    tagged_msg = stocktwits[stocktwits['sentiment'] != 'None']
    tagged_msg.reset_index(drop=True, inplace=True)
    # Pre process the messages
    tagged_msg['processed'] = tagged_msg['message'].apply(pre_process)
    tagged_msg.to_csv('data/02_processed/stocktwits_processed.csv')

    # Vectorized the processed message into features using TF-IDF method
    print("Vectorizing text messages using TF-IDF...")
    print("------------")
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(tagged_msg['processed'])
    vectorized_msg = pd.DataFrame.sparse.from_spmatrix(vectors)
    vectorized_msg.to_csv('data/03_vectorized/stocktwits_vectorized.csv')

    # Dimension Reduction using Truncated SVD
    print("Reducing dimensions using Truncated SVD...")
    print("------------")
    RANDOM_STATE = 42
    svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    features = svd.fit_transform(vectorized_msg)
    features = pd.DataFrame(features)
    print(
        f'Explained Variance Ratio: '
        f'{round(svd.explained_variance_ratio_.sum() * 100, 2)}%'
    )
    print("------------")

    # extract the index of messages tagged with Bull or Bear
    bear_id = tagged_msg['sentiment'][tagged_msg['sentiment'] == 'Bearish']
    bull_id = tagged_msg['sentiment'][tagged_msg['sentiment'] == 'Bullish']

    # randomly choose 50% bear messages for training set, the rest goes to
    # test set
    train_bear = bear_id.sample(frac=0.5, random_state=RANDOM_STATE)
    test_bear = bear_id.loc[bear_id.index.difference(train_bear.index)]

    # randomly choose bull messages for training set
    # (same len as bear messages for a balanced set)
    # the rest goes to test set
    train_bull = bull_id.sample(n=len(train_bear), random_state=RANDOM_STATE)
    test_bull = bull_id.loc[bull_id.index.difference(train_bull.index)]

    # final training & test set
    train_features = pd.concat(
        [features.loc[train_bear.index], features.loc[train_bull.index]])
    train_target = pd.concat([train_bear, train_bull])

    test_features = pd.concat(
        [features.loc[test_bear.index], features.loc[test_bull.index]])
    test_target = pd.concat([test_bear, test_bull])

    # apply Random Forest (base model)
    print("Applying Random Forest...")
    print("------------")

    rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                max_features='sqrt', n_jobs=-1,
                                max_samples=0.75, random_state=RANDOM_STATE,
                                oob_score=True)

    rf.fit(train_features, train_target)

    print("Evaluating the classifier's performance...")
    print("------------")


    def evaluate(model, test_features=test_features, test_target=test_target,
                 train_features=train_features, train_target=train_target):
        """
        Print confusion matrices and predictive accuracy
        (to evaluate the RF classfier's performance).
        Parameters
        ----------
        model : sklearn.ensemble._forest.RandomForestClassifier
            sklearn's Random Forest Classfier.
        test_features : DataFrame
            Features in test data.
        test_target : Series
            Target in test data.
        train_features : DataFrame
            Features in train data.
        train_target : Series
            Target in train data.
        Returns
        -------
        int
            Prediction accuracy (Test data)
        """
        test_pred = model.predict(test_features)
        train_pred = model.predict(train_features)
        test_cf = confusion_matrix(test_target, test_pred)
        train_cf = confusion_matrix(train_target, train_pred)

        accuracy = (test_cf[0][0] + test_cf[1][1]) / sum(sum(test_cf))

        cf_index = ['Bearish', 'Bullish']
        test_cf_df = pd.DataFrame(test_cf, columns=cf_index, index=cf_index)
        train_cf_df = pd.DataFrame(train_cf, columns=cf_index, index=cf_index)

        print('Model Performance')
        print(f'Accuracy: {round(accuracy * 100, 2)}%.')
        print("------------")
        print('Confusion Matrix (Test Data):')
        print(test_cf_df)
        print("------------")
        print('Confusion Matrix (Train Data):')
        print(train_cf_df)
        print("------------")

        return accuracy


    base_accuracy = evaluate(rf)

    # # Tuning parameters (Unfinished, blocked as comments)
    # from sklearn.model_selection import RandomizedSearchCV
    # random_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [5, 10, 20],
    #     'max_features': ['sqrt', 'log2'],
    #     'n_estimators': [50, 100, 200],
    #     'max_samples': [0.25, 0.5, 0.75]
    # }
    #
    # rf_tune = RandomForestClassifier()
    # rf_random = RandomizedSearchCV(estimator=rf_tune,
    #                                param_distributions=random_grid,
    #                                n_iter=25, cv=3,
    #                                random_state=RANDOM_STATE, n_jobs=-1)
    # rf_random.fit(train_features, train_target)
    # best_rf = rf_random.best_estimator_
    #
    # random_accuracy = evaluate(best_rf)
    #
    # impr_rate = (random_accuracy - base_accuracy) / base_accuracy
    # print(f"Improvement of {round(impr_rate * 100, 2)} %.")
