import pandas as pd
from pre_process import pre_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import warnings

# avoid FutureWarning (unsure future conficts between numpy and pandas)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # avoid false SettingWithCopyWarning

# import data
stocktwits = pd.read_csv("data/stocktwits.csv", index_col=0)
# extract all messages tagged with sentiment by user
tagged_msg = stocktwits[stocktwits['sentiment'] != 'None']
tagged_msg.reset_index(drop=True, inplace=True)
# pre process the messages  
tagged_msg['processed'] = tagged_msg['message'].apply(pre_process)

# vectorized the processed message into features using TF-IDF method
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(tagged_msg['processed'])
vectorized_msg = pd.DataFrame.sparse.from_spmatrix(vectors)

# Dimension Reduction using Truncated SVD
RANDOM_STATE = 42
svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
features = svd.fit_transform(vectorized_msg)
features = pd.DataFrame(features)
print(
    f'Explained Variance Ratio: '
    f'{round(svd.explained_variance_ratio_.sum() * 100, 2)}%'
)

# extract the index of messages tagged with Bull or Bear
bear_id = tagged_msg['sentiment'][tagged_msg['sentiment'] == 'Bearish']
bull_id = tagged_msg['sentiment'][tagged_msg['sentiment'] == 'Bullish']

# randomly choose 50% bear messages for training set, the rest goes to test set
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
rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                            max_features='sqrt', n_jobs=-1,
                            max_samples=0.75, random_state=RANDOM_STATE,
                            oob_score=True)

rf.fit(train_features, train_target)


# function to evaluate model:
def evaluate(model, testf=test_features, testt=test_target,
             trainf=train_features, traint=train_target):
    test_pred = model.predict(test_features)
    train_pred = model.predict(train_features)
    test_cf = confusion_matrix(test_target, test_pred)
    train_cf = confusion_matrix(train_target, train_pred)

    accuracy = (test_cf[0][0] + test_cf[1][1]) / sum(test_cf)

    cf_index = ['Bearish', 'Bullish']
    test_cf_df = pd.DataFrame(test_cf, columns=cf_index, index=cf_index)
    train_cf_df = pd.DataFrame(train_cf, columns=cf_index, index=cf_index)

    print('Model Performance')
    print(f'Accuracy: {round(accuracy * 100, 2)}%.')
    print('--------')
    print('Confusion Matrix (Test Data):')
    print(test_cf_df)
    print('--------')
    print('Confusion Matrix (Train Data):')
    print(train_cf_df)

    return accuracy


base_accuracy = evaluate(rf)

# tuning parameters
random_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [50, 100, 200],
    'max_samples': [0.25, 0.5, 0.75]
}

rf_tune = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf_tune,
                               param_distributions=random_grid,
                               n_iter=25, cv=3,
                               random_state=RANDOM_STATE, n_jobs=-1)
rf_random.fit(train_features, train_target)
best_rf = rf_random.best_estimator_

random_accuracy = evaluate(best_rf)

impr_rate = (random_accuracy - base_accuracy) / base_accuracy
print(f"Improvement of {round(impr_rate * 100, 2)} %.")
