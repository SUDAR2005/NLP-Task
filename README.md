## NLP Internship Task

This project implements various NLP techniques and machine learning models to classify emotions and predict sentiment from textual data. The tasks include data preprocessing, feature extraction, model building, and sentiment prediction using real-time Twitter data.

### Table of Contents

* [Dataset](#dataset)
* [Data Cleaning](#data-cleaning)
* [Data Preprocessing](#data-preprocessing)
* [Feature Extraction](#feature-extraction)
* [Model Building](#model-building)
* [Real-time Data Prediction](#real-time-data)
* [Visualization](#visualization)

### Dataset

The project uses the ISEAR dataset containing labeled emotional text data. Another dataset with Twitter data is used for real-time sentiment prediction.

### Data Cleaning

The data cleaning process involves:

* Converting text to lowercase
* Removing stopwords
* Stemming words

**Code Snippet:**

```python
# This code snippet is for illustration purposes only and might not reflect the actual implementation
data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
ps = SnowballStemmer(language='english')
data['text'] = data['text'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
```

### Data Preprocessing

Label encoding is used to convert emotion labels into numerical format.

**Code Snippet:**

```python
# This code snippet is for illustration purposes only and might not reflect the actual implementation
encoder = LabelEncoder()
data['emotion'] = encoder.fit_transform(data['emotion'])
```

### Feature Extraction

Features are extracted using CountVectorizer and TfidfVectorizer.

**Code Snippet:**

```python
# This code snippet is for illustration purposes only and might not reflect the actual implementation
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

rv = TfidfVectorizer()
X_train_rv = rv.fit_transform(X_train)
X_test_rv = rv.transform(X_test)
```

### Model Building

Several machine learning models are built and evaluated:

* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Logistic Regression

### Real-time Data Prediction

Real-time Twitter data is processed, and predictions are made using the trained models.

### Visualization

The results are visualized using pie charts for sentiment and emotion distributions.

**Explanation:**

You can define functions like `visualize_sentiment` and `visualize_emotion` to create pie charts using libraries like `matplotlib`.

### Conclusion

This project demonstrates using various NLP techniques and machine learning models for emotion and sentiment analysis from text data. It covers data preprocessing, feature extraction, model training, and result visualization.
