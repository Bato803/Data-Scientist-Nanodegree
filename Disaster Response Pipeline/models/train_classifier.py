import sys
# General libraries
import numpy as np
import pickle

# import NLP libs
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

# import ML libs
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table('clean_all', con=engine)
    
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    # url re expression, credit: https://www.geeksforgeeks.org/python-check-url-string/
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_re, text)
    
    # Replace urls in text with a string. Make it easier for the model to learn the pattern. 
    for url in urls:
        text.replace(url, 'url')
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words. 
    words = nltk.tokenize.word_tokenize(text)
    
    # Remove stop words. 
    words = [x for x in words if x not in stopwords.words('english')]
    
    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    cats = category_names
    y_pred = model.predict(X_test)
    y_true = Y_test.values
    
    print("{0:>30}\t\tPre\tRecall\tF1".format(""))
    
    pres, recalls, f1s = [], [], []
    
    for i, cat in enumerate(cats):
        pre = precision_score(y_true[:, i], y_pred[:, i], average="weighted")
        recall = recall_score(y_true[:, i], y_pred[:, i], average="weighted")
        f1 = f1_score(y_true[:, i], y_pred[:, i], average="weighted")
        
        pres.append(pre)
        recalls.append(pre)
        f1s.append(f1)
        
        print("{0:>30}\t\t{1:.3}\t{2:.3}\t{3:.3}".format(cat, pre, recall, f1))
    
    print("{0:>30}\t\t{1:.3}\t{2:.3}\t{3:.3}".format("Ave", np.mean(pres),
                                                     np.mean(recalls), np.mean(f1s)))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()