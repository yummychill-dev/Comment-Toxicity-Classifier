# Comment Toxicity Classifier

This project implements a **Comment Toxicity Classifier** using **BERT**, a state-of-the-art transformer model, to detect and classify toxic comments in text data. The model can classify comments as **toxic** or **non-toxic**, making it a powerful tool for online content moderation.

## 🚀 Features

- **Binary classification**: Identifies whether a comment is toxic or not.
- **BERT-based architecture**: Leverages the pre-trained `bert-base-uncased` model for sequence classification.
- **Scalable**: Capable of processing large datasets for robust performance.
- **High accuracy**: Evaluates performance using metrics such as **accuracy** during the training phase.
- **GPU support**: Utilizes GPU for faster training and inference when available.
- **Real-time predictions**: Quickly detects toxicity in comments.

## 🛠️ Tech Stack

- **Transformers**: For loading BERT tokenizer and model (`transformers` library by HuggingFace).
- **PyTorch**: For model training and inference.
- **Pandas & NumPy**: For data manipulation and processing.
- **Scikit-learn**: For splitting data into training and validation sets.
- **Evaluate**: To calculate accuracy during model evaluation.
- **AdamW Optimizer**: For effective gradient descent during training.

## 📁 Dataset

- The dataset used in this project contains comments labeled as **toxic** or **non-toxic**.
- The dataset is split into training and validation sets using an 80/20 ratio.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/toxic-comment-classifier.git
   cd toxic-comment-classifier
## 📊 Model Training

1. **Load Data**: The dataset is loaded using `pandas` and a random sample is taken for training and validation.
2. **Tokenization**: Comments are tokenized using BERT's `bert-base-uncased` tokenizer.
3. **Model Setup**: The BERT model is loaded for sequence classification and fine-tuned on the dataset.
4. **Training**: The model is trained using the HuggingFace `Trainer API` with an evaluation strategy for accuracy.
5. **Evaluation**: The trained model is evaluated on the validation set for accuracy.


## 🔄 Inference

To test the model on new comments:

## 🎯 Results

- The model is evaluated using **accuracy** as the metric, achieving robust performance on the validation set.
- Example result: `"I will kill you"` is classified as **toxic**.

## 🔧 Future Work

- **Multi-label classification**: Identifying specific types of toxicity (e.g., hate speech, threats).
- **Larger datasets**: Fine-tuning on larger datasets for improved accuracy.
- **Real-time integration**: Integrating the model with online platforms for real-time toxicity detection.

