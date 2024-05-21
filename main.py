import pandas as pd
import os
from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from nltk.corpus import wordnet
import random
import xgboost as xgb


# Reads the content of a text file and returns the content read as a string
def read_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Returns lists of labels contained within the provided file
def read_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        labels = [line.strip().split('\t') for line in file.readlines()]
    return labels

# Obtains embeddings for the given text using RoBERTa
def get_embeddings(text, tokenizer, model):
    """Obtains embeddings for the given text using RoBERTa."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()

# Augment text using synonyms from WordNet
def augment_text(text):
    augmented_text = []
    for word in text.split():
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            augmented_text.append(synonym)
        else:
            augmented_text.append(word)
    return ' '.join(augmented_text)


# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

data = []
train_folder_path = '/Users/saadelkadri/Downloads/Data/datasets-v5/tasks-2-3/train'

# Iterates over file reading the article and extracts segments of text from labels (if found)
for file_name in os.listdir(train_folder_path):
    if file_name.endswith('.txt'):
        article_path = os.path.join(train_folder_path, file_name)
        article_text = read_article(article_path)
        label_file_name = file_name.replace('.txt', '.task3.labels')
        label_path = os.path.join(train_folder_path, label_file_name)

        if os.path.exists(label_path):
            labels = read_labels(label_path)

        for label in labels:
            article_id, technique, startc, endc = label
            startc, endc = int(startc), int(endc)
            segment_text = article_text[startc:endc]

            embeddings = get_embeddings(segment_text, tokenizer, model)

            augmented_text = augment_text(segment_text)
            augmented_embeddings = get_embeddings(augmented_text, tokenizer, model)

            data.append([article_id, technique, startc, endc, segment_text, embeddings])
            data.append([article_id, technique, startc, endc, augmented_text, augmented_embeddings])

df = pd.DataFrame(data, columns=['article', 'label', 'startc', 'endc', 'text', 'embeddings'])

X = np.vstack(df['embeddings'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
y_train_numeric = y_train.map(label_mapping)
y_test_numeric = y_test.map(label_mapping)


xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_mapping), n_estimators=1000)
xgb_model.fit(X_train, y_train_numeric)
y_pred_xgb = xgb_model.predict(X_test)

print("Accuracy on original and augmented text:")
print(classification_report(y_test_numeric, y_pred_xgb, zero_division=0))
