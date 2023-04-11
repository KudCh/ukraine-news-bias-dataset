import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ernie import SentenceClassifier, Models

model = "intensified"

data = pd.read_csv("subjectivity_{}.csv".format(model))

#2 classes:
# < 1, >= 1
# 0: objective, 1: subjective
data["labels"] = np.where(data["scores"] < 1.0, 0, 1)

#--- preparing data for training---
X = pd.DataFrame(data.sentences)
y = pd.DataFrame(data.labels)
print(X)
print(y)

# set aside 20% of train and test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, shuffle = True, random_state = 8) # 20% for testing and validation



train_df = pd.DataFrame()
train_df["sentences"]=X_train["sentences"]
train_df["labels"]=y_train["labels"]

#--- Finetuning ---
classifier = SentenceClassifier(
    model_name=Models.BertBaseCased,
    max_length=64,
    labels_no=4
)
classifier.load_dataset(train_df, validation_split=0.2)
classifier.fine_tune(
    epochs=4,
    learning_rate=2e-5,
    training_batch_size=32,
    validation_batch_size=64
)

#--- Saving the model ---
classifier.dump('./model_{}'.format(model))

