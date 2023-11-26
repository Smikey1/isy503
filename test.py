from flask import Flask, render_template, request
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.utils import shuffle

from test import clean_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        sentence = request.form['sentence']  # Get user input from the form
        # Replace this with your actual sentiment analysis code
        analysis_result = perform_sentiment_analysis(sentence)

        # Display the sentiment analysis result on a new page or within the same page
        return render_template('result.html', result=analysis_result, sentence=sentence)


# Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)

    # Check the polarity of the sentiment
    if analysis.sentiment.polarity >= 0.5:
        return "Positive"
    elif analysis.sentiment.polarity < 0.5:
        return "Negative"
    else:
        return "Unreviewed"



def perform_sentiment_analysis(data_folder, model_directory, label_encoder_path, max_seq_length=128, batch_size=32, num_epochs=5, learning_rate=2e-5):
    # Task 1: Load the dataset and clean the data
    data, labels = load_and_clean_dataset(data_folder)

    # Task 2: Encode labels
    labels_encoded, label_encoder = encode_labels(labels)

    # Task 3: Remove outliers
    data, labels_encoded = remove_outliers(data, labels_encoded, min_length=10)

    # Task 4: Tokenize and pad data
    tokenizer = load_tokenizer()
    tokenized_data = tokenize_and_pad_data(data, tokenizer, max_seq_length)

    # Task 5: Split the data
    X_train, X_val, y_train, y_val, X_test, y_test = split_data(tokenized_data, labels_encoded)

    # Task 6: Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=batch_size)

    # Task 7: Define and instantiate the model
    model = define_and_instantiate_model(model_directory)

    # Task 8: Train the model
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate)

    # Task 9: Test the model
    test_loader = DataLoader(TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test), batch_size=batch_size)
    test_model(model, test_loader, label_encoder, label_encoder_path)

def load_and_clean_dataset(data_folder):
    # Initialize empty lists to store data and labels
    data = []
    labels = []

    # Define the categories you want to process
    categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']

    # Loop through selected categories
    for category in categories:
        category_folder = os.path.join(data_folder, category)
        # List all text files in the category folder
        for filename in os.listdir(category_folder):
            if filename.endswith('.review'):
                with open(os.path.join(category_folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = clean_text(text)  # Clean the text
                    data.append(cleaned_text)
                    labels.append(category)  # Use the folder name as the label

    return data, labels

# The remaining tasks (encode_labels, remove_outliers, tokenize_and_pad_data, split_data, create_data_loaders,
# define_and_instantiate_model, train_model, test_model) should be defined based on your specific implementation
# and choice of libraries/frameworks.


if __name__ == '__main__':
    app.run(debug=True)
