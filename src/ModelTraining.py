## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import logging
from utils import constant as const
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


models = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {"C": [0.01, 0.1, 1, 10], "max_iter": [100, 200, 300]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "params": {"alpha": [0.1, 0.5, 1.0]}
    }
}


def initiate_training( X_train_tfidf, X_test_tfidf, y_train, y_test):
    try:
        logging.info(f"Training the models")
        

        best_models = {}
        best_f1 = 0
        best_model_name = None
        models_accuracy = {}

        for name, config in models.items():
            print(f"\n🔍 Training {name}...")
    
            # Create model folder
            model_folder = os.path.join(const.MODEL_REPORTS, name.replace(" ", "_"))
            os.makedirs(model_folder, exist_ok=True)

            # Hyperparameter tuning
            grid_search = RandomizedSearchCV(config["model"], config["params"], cv=3, scoring="f1_weighted", verbose=1)
            grid_search.fit(X_train_tfidf, y_train)

            # Best model
            best_model = grid_search.best_estimator_
            best_models[name] = best_model

            # Save hyperparameters
            with open(f"{model_folder}/best_hyperparameters.txt", "w") as f:
                f.write(str(grid_search.best_params_))

            # Predictions
            y_train_pred = best_model.predict(X_train_tfidf)
            y_test_pred = best_model.predict(X_test_tfidf)

            # Compute Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            
            models_accuracy[name] = { 'Train Accuracy': train_acc, 'Test Accuracy': test_acc }
            logging.info(f"Model_accuracy: {models_accuracy[name]}")
            
            # Save classification report
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"{model_folder}/classification_report.csv")


            # Track best model based on F1-score
            f1_score = report["weighted avg"]["f1-score"]
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_name = name

            # Save the trained model
            joblib.dump(models_accuracy, f"{model_folder}/models_accuracy.txt")
            joblib.dump(best_model, f"{const.MODEL_OBJECTS}/best_model.pkl")
            logging.info(f"Model trained successfully: {name}")
            logging.info(f"\n✅ Best Model Selected: {best_model_name} with F1-score {best_f1}")
            
        return best_models[best_model_name]
            
    except Exception as e:  
        logging.error(f"Error in training the model: {e}")
        return None   


