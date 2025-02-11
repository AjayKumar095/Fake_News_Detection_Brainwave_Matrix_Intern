## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import nltk
import pandas as pd
import pickle as pkl
from utils.logger import logging
from nltk.corpus import stopwords
from utils import constant as const
from ETL_Pipeline.extract import extract_data
#from extract import extract_data
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")


class DataTransformer:
    
    def __init__(self):
        self.data = extract_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))


    def transform_data(self):
        try :
            self.data = self.__clean_data()
            if self.data is None:
                return None
            # Vectorize the text data
            logging.info("Transforming the data")
            X_train, X_test, y_train, y_test = train_test_split(self.data['text'],
                                                                self.data['traget'],
                                                                test_size=0.2, random_state=42)
            
            
            logging.info("Vectorizing the text data")
            X_train_TF = self.vectorizer.fit_transform(X_train)
            X_test_TF = self.vectorizer.transform(X_test)


            ## Save the vectorizer
            vectorizer_path = os.path.join(const.MODEL_OBJECTS, 'vectorizer.pkl')
            os.makedirs(const.MODEL_OBJECTS, exist_ok=True)
            with open(vectorizer_path, 'wb') as file:
                pkl.dump(self.vectorizer, file)

            
            logging.info("Data transformed successfully")
            return (X_train_TF, X_test_TF, y_train, y_test)
        
        except Exception as e:
            logging.error(f"Error in transforming the data: {e}")
            return None
        
    def __clean_data(self):
        try:
            logging.info("Cleaning the data")
            self.data.dropna(inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            # Apply regex to clean the text
            self.data['text'] = self.data['text'].apply(lambda x: re.sub("[^a-zA-z0-9]", ' ', str(x)))
            self.data['text'] = self.data['text'].apply(lambda x: x.lower())

            # Tokenize, remove stopwords, and lemmatize
            self.data['text'] = self.data['text'].apply(lambda x: ' '.join(
                [self.lemmatizer.lemmatize(word) for word in word_tokenize(x) if word not in self.stop_words]
            ))

            # Select only the 'text' and 'traget' columns
            self.data = self.data[['text', 'traget']]
            logging.info("Data cleaned successfully")
            return self.data
        except Exception as e:
            logging.error(f"Error in cleaning the data: {e}")
            return None    
        
