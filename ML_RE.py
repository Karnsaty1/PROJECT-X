import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Load the cleaned dataset
df = pd.read_csv('CleanResume.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['CleanedResume'], df['CategoryEncoded'], test_size=0.2, random_state=42
)

# Label encoding for the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Neural Network Model using Keras
model = Sequential()
model.add(Embedding(input_dim=X_train_tfidf.shape[1], output_dim=128, input_length=X_train_tfidf.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(df['CategoryEncoded'].unique()), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert sparse matrix to dense array for training
X_train_tfidf_dense = X_train_tfidf.toarray()

model.fit(X_train_tfidf_dense, y_train_encoded, epochs=5, batch_size=64, validation_split=0.2)

# Make predictions on the test set
X_test_tfidf_dense = X_test_tfidf.toarray()
y_pred_probabilities = model.predict(X_test_tfidf_dense)
y_pred_nn = y_pred_probabilities.argmax(axis=-1)

# Evaluate the Neural Network Model
accuracy_nn = accuracy_score(y_test_encoded, y_pred_nn)
classification_rep_nn = classification_report(y_test_encoded, y_pred_nn)

print(f"Neural Network Model Accuracy: {accuracy_nn}")
print("Neural Network Classification Report:\n", classification_rep_nn)

# Compare with other ML Models
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': MultinomialNB(),
    'XGBoost': XGBClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train_tfidf_dense, y_train_encoded)
    y_pred = clf.predict(X_test_tfidf_dense)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    classification_rep = classification_report(y_test_encoded, y_pred)
    
    print(f"\n{30 * '='}\n{name} Model Accuracy: {accuracy}")
    print(f"{name} Model Classification Report:\n", classification_rep)