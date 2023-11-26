from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Define the path to the 'positive.txt' and 'negative.txt' files
positive_file_path = 'data_source/positive.txt'
negative_file_path = 'data_source/negative.txt'

# Initialize variables for model and vectorizer
logistic_model = None
tfidf_vectorizer = None


# Helper function to read and preprocess the data
def read_data():
    data = []
    labels = []

    # Read positive reviews
    with open(positive_file_path, 'r', encoding='utf-8') as file:
        positive_reviews = file.readlines()
        data.extend(positive_reviews)
        labels.extend([1] * len(positive_reviews))  # Label 1 for positive

    # Read negative reviews
    with open(negative_file_path, 'r', encoding='utf-8') as file:
        negative_reviews = file.readlines()
        data.extend(negative_reviews)
        labels.extend([0] * len(negative_reviews))  # Label 0 for negative
    print(data)
    print(labels)
    return data, labels


# Function to load data, train the model, and make predictions
def analyze_sentiment(user_input):
    global logistic_model, tfidf_vectorizer

    if logistic_model is None or tfidf_vectorizer is None:
        # Initialize model and vectorizer if not already done
        data, labels = read_data()

        # Vectorize the text data using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_data = tfidf_vectorizer.fit_transform(data)

        # Split the data into training and testing sets
        X_train, _, y_train, _ = train_test_split(tfidf_data, labels, test_size=0.2, random_state=42)

        # Train a logistic regression model
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

    # Vectorize the user input
    user_input_vectorized = tfidf_vectorizer.transform([user_input])

    # Predict the sentiment
    sentiment_label = logistic_model.predict(user_input_vectorized)[0]
    # print()
    return 'Positive Review' if sentiment_label == 1 else f'Negative Review'


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment = analyze_sentiment(user_input)
    return render_template('index.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
