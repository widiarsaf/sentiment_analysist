# Import Library
from process_geneticAlgorithm import geneticAlgorithmProcess
from process_svm import svmProcess
from process_preprocessing import prepocessingText, preprocessingLocation
from process_weighting import splitDataset, tf_idf
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import re
import datetime as dt
import seaborn as sns
from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import csv
import os
import json
import warnings

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')


# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

# Library for Pre-processing
pd.options.mode.chained_assignment = None
seed = 0
np.random.seed(seed)
sns.set(style='whitegrid')
nltk.download('punkt')
nltk.download('stopwords')

# Import Methods

# Global Variable
train_X = []
test_X = []
train_Y = []
test_Y = []
train_x_arr = []
test_x_arr = []


# App
app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)
model = pickle.load(open("model.pkl", "rb"))

# Dataset
dataset = pd.read_csv("data_sentiment.csv")
datasets = dataset[['datetime', 'username', 'content', 'location', 'label']]
election_sentiment = pd.read_csv("sentiment_clean3.csv")

# Image Folder
imageFolder = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = imageFolder


# route to page
@app.route('/')
def home():
    images = "no_pict"
    return render_template('index.html', sentiment_image=images)


@app.route('/datasetPage')
def datasetPage():
    return render_template('data.html', tables=[datasets.to_html()], titles=[''])


@app.route('/chart')
def chart():
    # data_train = pd.read_csv("df_train.csv")
    # data_test = pd.read_csv("df_test.csv")
    # # images_chart_actual = os.path.join(app.config['UPLOAD_FOLDER'], 'Positive.png')
    # # images_chart_predict = os.path.join(app.config['UPLOAD_FOLDER'], 'Positive.png')

    # labels = ['Positive', 'Neutral', 'Negative']

    # pos_train = len(data_train[data_train["label"] == "positive"])
    # neu_train = len(data_train[data_train["label"] == "neutral"])
    # neg_train = len(data_train[data_train["label"] == "negative"])

    # category_train = [pos_train, neu_train, neg_train]

    # labels_json = json.dumps(labels)
    # category_json = json.dumps(category_train)

    # data_train_all = [labels_json, category_json]
    # data_train_json = json.dumps(data_train_all)
    # # print(data_train_all)

    # print(data)
    return render_template('chart.html')


@app.route('/preprocessingPage')
def preprocessingPage():
    return render_template('pre-processing.html')


@app.route('/weightingPage')
def weightingPage():
    return render_template('weighting.html')


@app.route('/svmPage')
def svmPage():
    return render_template('svm.html')


@app.route('/gaPage')
def gaPage():
    return render_template('ga.html')


# Methods
@app.route('/predict', methods=["POST"])
def predict():
    data = request.form["sentiment"]
    input_data = [data]
    tfidf_vect_data2 = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8',
                                       lowercase=True, max_df=1.0, max_features=30000, min_df=5,
                                       ngram_range=(1, 1), norm='l2',
                                       strip_accents='unicode', sublinear_tf=False,
                                       token_pattern='\\w{1,}', tokenizer=None, use_idf=True,
                                       vocabulary=None)
    tfidf_vect_data2.fit(election_sentiment['sentiment'])
    vectorized_input = tfidf_vect_data2.transform(input_data)
    new_test_x2 = vectorized_input.toarray()
    prediction = model.predict(new_test_x2)
    predict_proba = model.predict_proba(new_test_x2)
    positive = predict_proba[0][2]
    neutral = predict_proba[0][1]
    negative = predict_proba[0][0]
    predict_text = ''
    images = "no_pict"
    data2 = request.form.get('sentiment')

    if prediction == ['positive']:
        predict_text = "Positive"
        images = os.path.join(app.config['UPLOAD_FOLDER'], 'Positive.png')

    elif prediction == ['neutral']:
        predict_text = "Neutral"
        images = os.path.join(app.config['UPLOAD_FOLDER'], 'Neutral.png')

    elif prediction == ['negative']:
        predict_text = "Negative"
        images = os.path.join(app.config['UPLOAD_FOLDER'], 'Negative.png')

    return render_template("index.html", sentiment_image=images, inputData=data2, prediction_text="{}".format(predict_text),
        positive = positive, neutral = neutral, negative = negative
        )


# Preprocessing
@app.route('/preprocessing', methods=["POST"])
def preprocessing():
    datasets['locations'] = preprocessingLocation(datasets['location'])
    datasets['sentiment'] = prepocessingText(datasets['content'])
    datasets['labels'] = datasets['label']
    datasets.to_csv("data_sentiment_clean.csv", mode='w', index=False)

    sentiment_clean = pd.read_csv("data_sentiment_clean.csv")
    sentiment_clean[['datetime', 'username', 'content',
                     'location', 'label', 'sentiment', 'locations', 'labels']]

    # show_data = np.array([datasets['content'],sentiment_clean['sentiment']])
    show_data_df = pd.DataFrame(
        [datasets['content'], sentiment_clean['sentiment']], index=['Before Preprocessing', 'After Preprocessing']).T

    # drop column
    datasets.drop(['sentiment'], axis=1, inplace=True)
    datasets.drop(['locations'], axis=1, inplace=True)
    datasets.drop(['labels'], axis=1, inplace=True)
    sentiment_clean.drop(['username'], axis=1, inplace=True)
    sentiment_clean.drop(['content'], axis=1, inplace=True)
    sentiment_clean.drop(['location'], axis=1, inplace=True)
    sentiment_clean.drop(['label'], axis=1, inplace=True)
    sentiment_clean.to_csv("data_sentiment_clean.csv", mode='w', index=False)

    return render_template('pre-processing.html', tables=[show_data_df.to_html()], titles=[''], done="OK!")


@app.route('/splitData', methods=["POST"])
def splitData():
    global train_X, test_X, train_Y, test_Y
    sentiment_clean = pd.read_csv("data_sentiment_clean.csv")
    df_train, df_test,  train_X, test_X, train_Y, test_Y = splitDataset(
        sentiment_clean['sentiment'], sentiment_clean['labels'])

    # save data train and dataset
    df_train.to_csv("df_train.csv", mode='w', index=False)
    df_test.to_csv("df_test.csv", mode='w', index=False)
    total_train = len(df_train)
    total_test = len(df_test)
    return render_template('weighting.html', total_train=total_train, total_test=total_test)


@app.route('/weighting', methods=["POST"])
def weighting():
    global train_x_arr
    global test_x_arr
    sentiment_clean = pd.read_csv("data_Sentiment_clean.csv")
    df_train = pd.read_csv("df_train.csv")
    df_test = pd.read_csv("df_test.csv")
    test_x_arr, train_x_arr, train_shape, test_shape = tf_idf(
        df_train, 	df_test, sentiment_clean)
    total_train = len(df_train)
    total_test = len(df_test)
    return render_template('weighting.html', done="OK! TF-IDF Done", total_train=total_train, total_test=total_test, train_shape=train_shape, test_shape=test_shape)


@app.route('/svm_process', methods=["POST"])
def svm_process():
    warnings.filterwarnings('ignore')
    accuracy, classification_report, summarized_report, confusion_matrix_report = svmProcess(
        train_x_arr, test_x_arr, train_Y, test_X, test_Y)

    # Plot the confusion matrix.
    img = BytesIO()
    sns.heatmap(confusion_matrix_report, annot=True)
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('svm.html', accuracy=accuracy, summarized_reports=[summarized_report.to_html()], titles=[''], images=plot_url)


@app.route('/geneticAlgorithm', methods=["POST"])
def geneticAlgorithm():
    warnings.filterwarnings('ignore')

    #get data
    population = int(request.form["population"])
    crossover = float(request.form["crossover"])
    mutation = float(request.form["mutation"])
    generation = int(request.form["generation"])
    log, best_param, plot_ga = geneticAlgorithmProcess(
        train_x_arr, test_x_arr, train_Y, test_Y, population, crossover,mutation,generation)
    log_df = pd.DataFrame(log)
   
    c = best_param[0]
    kernel = best_param[1]
    degree = best_param[2]
    gamma = best_param[3]
    coef0 = best_param[4]
    max_iter = best_param[5]
    
    #Plot
    img = BytesIO()
    plot_ga
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_ga= base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('ga.html', done="Ok!", logs=[log_df.to_html()], titles=[''], best_param=best_param, 
                           images=plot_ga, c=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, max_iter=max_iter)


if __name__ == "__main__":
    app.run(debug=True)
