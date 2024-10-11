import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import evaluate  


# Load the dataset
a = pd.read_csv(r"C:\Users\toyas\Downloads\Toxicity Classifier\DataSet_2\train.csv")
# Assuming your DataFrame is named 'df'
df = a.sample(frac=0.1, random_state=42)

# Prepare the data
comments = df["comment_text"].values
labels = df["toxic"].values  # Using "toxic" column for binary classification (0 or 1)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(comments, labels, test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(
        list(texts),  # Convert texts to a list to ensure proper format
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Prepare dataset for PyTorch
class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenize_data(texts, tokenizer)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = ToxicCommentsDataset(train_texts, train_labels, tokenizer)
val_dataset = ToxicCommentsDataset(val_texts, val_labels, tokenizer)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a function to compute accuracy using evaluate
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")  # Use evaluate to load the accuracy metric
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get predicted labels
    return metric.compute(predictions=predictions, references=labels)  # Compute accuracy


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    push_to_hub=False,
    fp16=torch.cuda.is_available()  # Use mixed precision if GPU is available for faster training
)



# Create Trainer instance with the compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # NEW: Add compute_metrics to calculate accuracy
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the trained model
model.save_pretrained("./toxic_comment_model")
tokenizer.save_pretrained("./toxic_comment_model")


# -----------------------------------------------

#Testing of the model 
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("./toxic_comment_model")
model = BertForSequenceClassification.from_pretrained("./toxic_comment_model")

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the input text (the comment you want to evaluate for toxicity)
input_text = "I will kill you"  # Replace this with your desired comment

# Tokenize the input text
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)

# Move the tokenized data to the appropriate device
inputs = {key: val.to(device) for key, val in inputs.items()}

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted label (0 for non-toxic, 1 for toxic)
predicted_label = torch.argmax(logits, dim=1).item()

# Print the result
if predicted_label == 1:
    print("The comment is toxic.")
else:
    print("The comment is not toxic.")
    
    
    
    
    
    
    
    
    
    

