# Import necessary libraries and modules
import os
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from textblob import TextBlob  # Import TextBlob for sentiment analysis
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model_class import SentimentClassifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        sentence = request.form['sentence']  # Get user input from the form
        # Task 10: Perform sentiment analysis and return the result
        analysis_result = perform_sentiment_analysis(sentence)

        # Display the sentiment analysis result on a new page or within the same page
        return render_template('result.html', result=analysis_result, sentence=sentence)

# Task 1: Data Loading and Preprocessing
def load_and_preprocess_data(data_folder='sorted_data_acl'):
    data = []
    labels = []

    # Define the categories you want to process
    categories = os.listdir(data_folder)

    # Loop through selected categories
    for category in categories:
        category_folder = os.path.join(data_folder, category)

        # List all text files in the category folder
        for filename in os.listdir(category_folder):
            if filename.endswith('.review'):
                with open(os.path.join(category_folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Task 1: Data Cleaning (Perform data cleaning logic here, e.g., removing punctuation)
                    cleaned_text = clean_text(text)
                    data.append(cleaned_text)
                    labels.append(category)  # Use the folder name as the label

    # Encode labels (convert text labels to numerical values)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Shuffle and split the data into training and validation sets
    data_shuffled, labels_encoded_shuffled = shuffle(data, labels_encoded, random_state=42)
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        data_shuffled, labels_encoded_shuffled, test_size=0.2, random_state=42
    )

    # Perform any additional data preprocessing steps as needed

    return train_data, train_labels, valid_data, valid_labels

# Define a function to clean the text (implement your specific cleaning logic)
def clean_text(text):
    # Implement your text cleaning logic here
    return text


# Define the SentimentClassifier class (Task 8)
def train_and_test_model(model, train_data, train_labels, validation_data, validation_labels, test_data, test_labels, num_epochs=3, batch_size=32, learning_rate=1e-5):
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels)
    validation_data = torch.tensor(validation_data)
    validation_labels = torch.tensor(validation_labels)
    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)

    # Create DataLoader for training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate and print average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {average_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        validation_outputs = model(validation_data)
        _, predicted = torch.max(validation_outputs, 1)
        total += validation_labels.size(0)
        correct += (predicted == validation_labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_outputs = model(test_data)
        _, predicted = torch.max(test_outputs, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate and print classification report
    test_labels = test_labels.numpy()
    predicted = predicted.numpy()
    report = classification_report(test_labels, predicted, target_names=["Negative", "Positive"])
    print(report)

# Task 10: Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)

    # Check the polarity of the sentiment
    if analysis.sentiment.polarity >= 0.5:
        return "Positive"
    elif analysis.sentiment.polarity < 0.5:
        return "Negative"
    else:
        return "Unreviewed"

# Task 2: Data Encoding (Encode the words in the review)

def encode_data(data, tokenizer, max_seq_length):
    tokenized_data = []

    for text in data:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, truncation=True)

        # Pad or truncate the tokenized sequence to the specified length
        if len(tokens) < max_seq_length:
            tokens += [tokenizer.pad_token_id] * (max_seq_length - len(tokens))
        else:
            tokens = tokens[:max_seq_length]

        # Convert the tokenized sequence to a PyTorch tensor
        tokenized_data.append(torch.tensor(tokens))

    return tokenized_data

# Task 3: Encode the labels for 'positive' and 'negative'
def encode_labels(labels):
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the labels and transform them to numerical values
    labels_encoded = label_encoder.fit_transform(labels)

    return labels_encoded, label_encoder

# Task 4: Conduct outlier removal to eliminate really short or wrong reviews
def remove_outliers(data, labels, min_length=10):
    # Initialize lists to store cleaned data and labels
    cleaned_data = []
    cleaned_labels = []

    for review, label in zip(data, labels):
        # Check if the review meets the minimum length criteria
        if len(review.split()) >= min_length:
            cleaned_data.append(review)
            cleaned_labels.append(label)

    return cleaned_data, cleaned_labels

# Task 5: Pad/truncate remaining data
def pad_truncate_data(data, tokenizer, max_seq_length):
    tokenized_data = []

    for text in data:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, truncation=True)

        # Pad or truncate the tokenized sequence to the specified length
        if len(tokens) < max_seq_length:
            tokens += [tokenizer.pad_token_id] * (max_seq_length - len(tokens))
        else:
            tokens = tokens[:max_seq_length]

        # Convert the tokenized sequence to a PyTorch tensor
        tokenized_data.append(torch.tensor(tokens))

    return tokenized_data

# Task 6: Split the data into training, validation, and test sets
def split_data(data, labels, test_size=0.2, validation_size=0.2, random_state=42):
    # Split the data and labels into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state)

    # Further split the training data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_size, random_state=random_state)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# Task 7: Define the network architecture (This part can be defined in the model_class.py file)
# Task 8: Define the model class (This part can be defined in the model_class.py file)


if __name__ == "__main__":
    app.run(debug=True)
