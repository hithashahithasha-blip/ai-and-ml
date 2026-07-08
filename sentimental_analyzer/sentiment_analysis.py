import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Cleans and preprocesses the text for analysis.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize (simple split by whitespace)
    words = text.split()
    
    # 4. Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def main():
    print("1. Loading dataset...")
    # Loading the local IMDB Dataset CSV file
    df = pd.read_csv('IMDB Dataset.csv.zip')
    
    # Map 'positive' to 1 and 'negative' to 0 to work with the prediction logic below
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Optional: The dataset contains 50,000 records. Preprocessing all of them might take a few minutes.
    # If you want a quicker run, uncomment the following line to use a subset of the data:
    # df = df.sample(n=2000, random_state=42)
    
    print("2. Preprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    print("3. Splitting into Train and Test sets...")
    X = df['cleaned_review']
    y = df['sentiment']
    # 75% for training, 25% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("4. Feature Extraction (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Only transform the test data to prevent data leakage
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("5. Training Machine Learning Model...")
    # Using Logistic Regression as it works well for binary classification text tasks
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    print("6. Evaluating the Model...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    # Contains Precision, Recall, and F1-Score
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n7. Making Predictions on new data...")
    new_reviews = [
        "I really enjoyed the storyline and the characters.",
        "This was absolutely disgusting and boring."
    ]
    
    # Preprocess and vectorize the new reviews before predicting
    cleaned_new_reviews = [preprocess_text(rev) for rev in new_reviews]
    new_reviews_tfidf = vectorizer.transform(cleaned_new_reviews)
    
    # Make predictions
    predictions = model.predict(new_reviews_tfidf)
    
    print("\nPredictions for new reviews:")
    for rev, pred in zip(new_reviews, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Review: '{rev}' \n-> Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()