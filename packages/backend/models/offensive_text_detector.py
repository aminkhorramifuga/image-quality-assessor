from transformers import BertForSequenceClassification

class OffensiveTextDetector:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def predict(self, text):
        # You would need to preprocess the text into the format expected by BERT,
        # then feed it into the model and interpret the output.
        pass
