import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from textblob import TextBlob  # Import TextBlob for sentiment analysis
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import BertTokenizer
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


def load_and_clean_dataset(data_folder):
    data = []
    labels = []
    categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']

    for category in categories:
        category_folder = os.path.join(data_folder, category)
        for filename in os.listdir(category_folder):
            if filename.endswith('.review'):
                with open(os.path.join(category_folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Perform data cleaning here (e.g., removing punctuation, lowercasing)
                    cleaned_text = clean_text(text)
                    data.append(cleaned_text)
                    labels.append(category)  # Use the folder name as the label

    # Create a DataFrame for easy manipulation
    df = pd.DataFrame({'text': data, 'label': labels})

    # Remove outliers based on text length
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['text_length'] >= min_length]

    return df['text'].tolist(), df['label'].tolist()

@app.route('/load_and_clean_data')
# Define a function to clean the text (implement your specific cleaning logic)
def clean_text(text):
    # Implement your text cleaning logic here
    return text

# Set the data folder path
data_folder = 'sorted_data_acl'

# Load and clean the dataset
data, labels = load_and_clean_dataset(data_folder)



@app.route('/encode_label')
def encode_labels(labels):
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the labels and transform them to numerical values
    labels_encoded = label_encoder.fit_transform(labels)

    return labels_encoded, label_encoder

# Example labels (replace with your actual labels)
labels = ["positive", "negative", "positive", "negative"]

# Encode the labels
labels_encoded, label_encoder = encode_labels(labels)

# Now, labels_encoded contains numerical representations of the labels, and label_encoder
# can be used to map them back to their original text labels.

@app.route('/encode_sentiment_labels')
def encode_sentiment_labels(labels):
    # Define a mapping dictionary for 'positive' and 'negative' labels
    label_mapping = {'positive': 1, 'negative': 0}

    # Use the mapping to encode the labels
    encoded_labels = [label_mapping[label] for label in labels]

    return encoded_labels


# Example sentiment labels (replace with your actual labels)
sentiment_labels = ['positive', 'negative', 'positive', 'negative']

# Encode the sentiment labels
encoded_sentiment_labels = encode_sentiment_labels(sentiment_labels)

# Now, encoded_sentiment_labels contains numerical representations of the sentiment labels.

@app.route('/remove_outliers')
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

# Example data and labels (replace with your actual data)
data = ["This is a short review.", "This is a longer review with more content.", "Invalid data."]
labels = ["positive", "negative", "negative"]

# Set a minimum length threshold (adjust as needed)
min_length_threshold = 5

# Remove outliers based on the minimum length
cleaned_data, cleaned_labels = remove_outliers(data, labels, min_length_threshold)

# Now, cleaned_data and cleaned_labels contain reviews and labels after outlier removal.



@app.route('/tokenize_and_pad_data')
def tokenize_and_pad_data(data, tokenizer, max_seq_length):
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


# Example variables (replace with your actual data)
data = ["This is a sample review.", "Another review for testing."]
max_seq_length = 128  # Maximum sequence length for padding

# Load a BERT tokenizer (replace 'bert-base-uncased' with your desired BERT model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad the data
tokenized_data = tokenize_and_pad_data(data, tokenizer, max_seq_length)



@app.route('split_data_after_tokenization')
def split_data_after_tokenization(tokenized_data, labels, test_size=0.2, validation_size=0.2, random_state=42):
    # Split the tokenized data and labels into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        tokenized_data, labels, test_size=test_size, random_state=random_state)

    # Further split the training data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_size, random_state=random_state)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# Example variables (replace with your actual data)
tokenized_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
labels = ["positive", "negative"]

# Split the data using the function
train_data, val_data, test_data, train_labels, val_labels, test_labels = split_data_after_tokenization(
    tokenized_data, labels)


@app.route('/get_training_dataloader')
def get_training_dataloader(train_data, train_labels, batch_size=32, shuffle=True):
    """
    Create a DataLoader for training data.

    Args:
    train_data (list of torch.Tensor): List of training data samples.
    train_labels (list or torch.Tensor): List or tensor of corresponding training labels.
    batch_size (int): Batch size for the DataLoader (default is 32).
    shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
    DataLoader: DataLoader for training data.
    """
    # Convert the training data and labels to PyTorch tensors
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    # Create a TensorDataset to combine data and labels
    train_dataset = TensorDataset(train_data, train_labels)

    # Create a DataLoader for training data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader

# Example variables (replace with your actual data)
train_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
train_labels = ["positive", "negative", "positive"]

# Set batch size
batch_size = 2

# Create a DataLoader for training data
train_dataloader = get_training_dataloader(train_data, train_labels, batch_size=batch_size, shuffle=True)


@app.route('/train_model')
def train_model(model, train_data, train_labels, validation_data, validation_labels, num_epochs=3, batch_size=32, learning_rate=1e-5):
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels)
    validation_data = torch.tensor(validation_data)
    validation_labels = torch.tensor(validation_labels)

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



def load_and_preprocess_data(data_folder):
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
                    data.append(text)
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



# Define the SentimentClassifier class here

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


# Example usage:
if __name__ == "__main__":
    app.run(debug=True)
    data_folder = 'sorted_data_acl'  # Replace with the path to your data folder
    train_reviews, train_labels, valid_reviews, valid_labels =  load_and_preprocess_data(data_folder)

    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    model = SentimentClassifier(bert_model_name, num_classes)

    # Load and preprocess your training, validation, and test data
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels =  load_and_preprocess_data(data_folder)

    # Train and test the model
    train_and_test_model(model, train_data, train_labels, validation_data, validation_labels, test_data, test_labels)

    # Now you can use train_reviews, train_labels, valid_reviews, and valid_labels
    # in the train_model function or any other part of your code as needed.