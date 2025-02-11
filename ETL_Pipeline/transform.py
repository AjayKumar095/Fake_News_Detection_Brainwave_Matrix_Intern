## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import nltk
import pandas as pd
from utils import logger
from extract import extract_data
from nltk.corpus import stopwords
from utils import constant as const
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
    
    def __init__(self,data):
        self.data = data
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()


    def transform_data(self):
        try :
            logger.info("Transforming the data")
            self.data = self.__clean_data()
            if self.data is None:
                return None
            # Vectorize the text data
            
            X_train, X_test, y_train, y_test = train_test_split(self.data['text'],
                                                                self.data['label'],
                                                                test_size=0.2, random_state=42)
            
            
            logger.info("Vectorizing the text data")
            X_train_TF = self.vectorizer.fit_transform(X_train)
            X_test_TF = self.vectorizer.transform(X_test)
            
            training_data = pd.concat([pd.DataFrame(X_train_TF.toarray()), y_train.reset_index(drop=True)], axis=1)
            test_data = pd.concat([pd.DataFrame(X_test_TF.toarray()), y_test.reset_index(drop=True)], axis=1)
            
            training_data.to_csv(const.TRAINING_DATA_PATH, index=False)
            test_data.to_csv(const.TEST_DATA_PATH, index=False)
            
            logger.info("Data transformed successfully")
            return (const.TRAINING_DATA_PATH, const.TEST_DATA_PATH)
        
        except Exception as e:
            logger.error(f"Error in transforming the data: {e}")
            return None
        
    def __clean_data(self):
        try:
            logger.info("Cleaning the data")
            self.data.dropna(inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            # Apply regex to clean the text
            self.data['text'] = self.data['text'].apply(lambda x: re.sub("[^a-zA-z0-9]", ' ', str(x)))
            self.data['text'] = self.data['text'].apply(lambda x: x.lower())

            # Tokenize, remove stopwords, and lemmatize
            self.data['text'] = self.data['text'].apply(lambda x: ' '.join(
                [self.lemmatizer.lemmatize(word) for word in word_tokenize(x) if word not in self.stop_words]
            ))

            # Select only the 'text' and 'label' columns
            self.data = self.data[['text', 'label']]
            logger.info("Data cleaned successfully")
            return self.data
        except Exception as e:
            logger.error(f"Error in cleaning the data: {e}")
            return None    
        
