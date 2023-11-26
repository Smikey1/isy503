import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(SentimentClassifier, self).__init__()

        # Task 7: Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Task 7: Define a custom classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),  # Adjust the hidden size as needed
            nn.ReLU(),
            nn.Dropout(0.2),  # Adjust the dropout rate as needed
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Pass the input through the BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Get the BERT model's output representation
        pooled_output = outputs.pooler_output

        # Pass the representation through the custom classifier
        logits = self.classifier(pooled_output)

        return logits

