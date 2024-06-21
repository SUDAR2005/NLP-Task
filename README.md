# NLP-Task
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Internship Task</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }

        h1, h2, h3 {
            color: #0056b3;
        }

        pre {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }

        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 4px;
        }

        .container {
            max-width: 800px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>NLP Internship Task</h1>
        <p>This project is an implementation of various NLP techniques and machine learning models to classify emotions and predict sentiment from textual data. The task includes data preprocessing, feature extraction, model building, and sentiment prediction using real-time Twitter data.</p>

        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#dataset">Dataset</a></li>
            <li><a href="#data-cleaning">Data Cleaning</a></li>
            <li><a href="#data-preprocessing">Data Preprocessing</a></li>
            <li><a href="#feature-extraction">Feature Extraction</a></li>
            <li><a href="#model-building">Model Building</a></li>
            <li><a href="#real-time-data">Real-time Data Prediction</a></li>
            <li><a href="#visualization">Visualization</a></li>
        </ul>

        <h2 id="dataset">Dataset</h2>
        <p>The dataset used in this project is the ISEAR dataset, which contains textual data labeled with different emotions.</p>
        <p> Another dataset containing Twitter data is used for real-time sentiment prediction.</p>

        <h2 id="data-cleaning">Data Cleaning</h2>
        <p>The data cleaning process includes:</p>
        <ul>
            <li>Converting text to lowercase</li>
            <li>Removing stopwords</li>
            <li>Stemming to obtain root words</li>
        </ul>
        <pre><code>data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
ps = SnowballStemmer(language='english')
data['text'] = data['text'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))</code></pre>

        <h2 id="data-preprocessing">Data Preprocessing</h2>
        <p>Label encoding is used to convert categorical labels into numeric format.</p>
        <pre><code>encoder = LabelEncoder()
data['emotion'] = encoder.fit_transform(data['emotion'])</code></pre>

        <h2 id="feature-extraction">Feature Extraction</h2>
        <p>Features are extracted using CountVectorizer and TfidfVectorizer.</p>
        <pre><code>cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

rv = TfidfVectorizer()
X_train_rv = rv.fit_transform(X_train)
X_test_rv = rv.transform(X_test)</code></pre>

        <h2 id="model-building">Model Building</h2>
        <p>Several machine learning models are built and evaluated, including:</p>
        <ul>
            <li>Naive Bayes</li>
            <li>Support Vector Machine (SVM)</li>
            <li>Random Forest</li>
            <li>Logistic Regression</li>
        </ul>
        <pre><code># Naive Bayes model
naive_bayes_model = MultinomialNB()
model1 = build(naive_bayes_model, X_train_cv, y_train, X_test_cv)

# SVM model
svm_model = SVC()
model3 = build(svm_model, X_train_cv, y_train, X_test_cv)

# Random Forest model
rf_model = RandomForestClassifier()
model5 = build(rf_model, X_train_cv, y_train, X_test_cv)

# Logistic Regression model
lr_model = LogisticRegression()
model7 = build(lr_model, X_train_cv, y_train, X_test_cv)</code></pre>

        <h2 id="real-time-data">Real-time Data Prediction</h2>
        <p>The real-time data from Twitter is processed and predictions are made using the trained models.</p>
        <pre><code>test_pd['text'] = test_pd['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
test_pd['text'] = test_pd['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test_pd['text'] = test_pd['text'].apply(clean_text)
predict_cv = cv.transform(test_pd['text'])

# Prediction using Naive Bayes model
make_prediction(model1, predict_cv)</code></pre>

        <h2 id="visualization">Visualization</h2>
        <p>The results are visualized using pie charts for sentiment and emotion distributions.</p>
        <pre><code>def visualize_sentiment(Sentiment_df):
    fig, ax = plt.subplots()
    ax.pie(Sentiment_df['Count'], labels=Sentiment_df['Sentiment'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Sentiment Analysis')
    plt.show()

def visualize_emotion(Emotion_df):
    fig, ax = plt.subplots()
    ax.pie(Emotion_df['Count'], labels=Emotion_df['Emotion'], autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'purple', 'orange', 'blue', 'yellow', 'pink'], pctdistance=0.85)
    ax.axis('equal')
    plt.title('Emotion Analysis')
    plt.show()</code></pre>

        <h2>Conclusion</h2>
        <p>This project demonstrates the use of variousmachine learning models to analyze and predict emotions and sentiments from text data. The implementation covers data preprocessing, feature extraction, model training and visualization of results.</p>
    </div>
</body>

</html>
