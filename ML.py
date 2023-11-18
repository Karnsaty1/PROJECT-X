import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('CleanResume.csv')

# Load the label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = df['CategoryEncoded'].unique()

# Load the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(df['CleanedResume'])

# Load the Neural Network Model
from ML_RE import model as nn_model  # Change 'my' to the actual module name

# Load other ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
svc_model = SVC()
nb_model = MultinomialNB()
xgb_model = XGBClassifier()

# Load pre-trained weights for other ML models
rf_model.fit(X_train_tfidf, label_encoder.transform(df['CategoryEncoded']))
gb_model.fit(X_train_tfidf, label_encoder.transform(df['CategoryEncoded']))
svc_model.fit(X_train_tfidf, label_encoder.transform(df['CategoryEncoded']))
nb_model.fit(X_train_tfidf, label_encoder.transform(df['CategoryEncoded']))
# xgb_model.fit(X_train_tfidf, label_encoder.transform(df['CategoryEncoded']))

# Text preprocessing function
def preprocess_input(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=X_train_tfidf.shape[1], padding='post')
    return padded_sequences

# Get user input
user_input = input("Enter a resume content: ")

# Preprocess the input
preprocessed_input = preprocess_input(user_input)

# Make predictions using the Neural Network Model
nn_probabilities = nn_model.predict(preprocessed_input)
nn_confidence = np.max(nn_probabilities) * 100
nn_prediction = label_encoder.inverse_transform(nn_probabilities.argmax(axis=-1))[0]
print(f"Neural Network Prediction: {nn_prediction} (Confidence: {nn_confidence:.2f}%)")

# Make predictions using other ML models
rf_confidence = np.max(rf_model.predict_proba(X_train_tfidf)) * 100
gb_confidence = np.max(gb_model.predict_proba(X_train_tfidf)) * 100
svc_confidence = np.max(svc_model.predict_proba(X_train_tfidf)) * 100
nb_confidence = np.max(nb_model.predict_proba(X_train_tfidf)) * 100
xgb_confidence = np.max(xgb_model.predict_proba(X_train_tfidf)) * 100

# Get the most confident prediction among other models
confidences = {
    'Random Forest': rf_confidence,
    'Gradient Boosting': gb_confidence,
    'Support Vector Machine': svc_confidence,
    'Naive Bayes': nb_confidence,
    'XGBoost': xgb_confidence
}

most_confident_model = max(confidences, key=confidences.get)
most_confident_confidence = confidences[most_confident_model]

print(f"Most Confident Model: {most_confident_model} (Confidence: {most_confident_confidence:.2f}%)")
