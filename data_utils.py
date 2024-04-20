import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


def load_data():
    train_data = pd.read_csv(r'data\SMS_train.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv(r'data\SMS_test.csv', encoding='ISO-8859-1')
    return train_data, test_data


def preprocess_data(train_data, test_data):
    X_train = train_data['Message_body']
    y_train = train_data['Label']
    X_test = test_data['Message_body']
    y_test = test_data['Label']

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
    vectorizer.fit_transform(X_train.tolist())
    vectorizer.transform(X_test.tolist())

    return X_train, y_train, X_test, y_test, vectorizer
