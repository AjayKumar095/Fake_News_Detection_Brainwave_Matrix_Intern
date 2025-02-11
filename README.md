# Fake News Classification Project

## Overview
This project is designed to classify news articles as **Fake** or **Real** using various machine learning models. The dataset is processed using **TF-IDF** vectorization, and multiple models including **Naïve Bayes (NB)**, **Random Forest (RF)**, and **Logistic Regression (LR)** are trained. The best model is selected based on performance metrics.

The project includes:
- **Custom ETL Pipeline**: Data cleaning, preprocessing (lemmatization & stopword removal using NLTK).
- **Model Training & Evaluation**: Comparing multiple models to choose the best.
- **Flask-based Web Interface**: User-friendly UI for news classification.
- **Custom Logger**: Logging for debugging and tracking the process.

## Dataset
- Download the dataset from below link and place the dataset file in Artifacts/Dataset folder and make changes in file path name in utils/constant.py file if needed.
- **Dataset Link**: [Fake News Dataset](https://www.kaggle.com/datasets/ajaykumar0090/fake-news-detection-modified)



## Tools & Technologies
- **Programming Language**: Python
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **NLP Tools**: NLTK (Lemmatization & Stopword Removal)
- **Vectorization**: TF-IDF
- **Flask**: Web application
- **Logging**: Custom logger for debugging

## Installation & Running Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/AjayKumar095/Fake_News_Detection_Brainwave_Matrix_Intern.git
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python trainer.py
```

### Step 4: Run Flask App
```bash
python app.py
```

### Step 5: Open in Browser
Go to: `http://127.0.0.1:5000/`

## Project Screenshots
![Flask UI](/Artifacts/ProjectDocs/flaskUI.png)

## Author
- **Ajay Kumar**  
- **GitHub**: [GitHub Profile](https://github.com/AjayKumar095/)
- **Linkedin**: [Linkedin profile](https://www.linkedin.com/in/ajay-kumar-4b1b7329a/)
