import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ernie import SentenceClassifier, Models

dataset = sys.argv[1]
data = pd.read_csv("subjectivity_{}.csv".format(dataset))

# 2 classes:
# < 1, >= 1
# 0: objective, 1: subjective
data["labels"] = np.where(data["scores"] < 1.0, 0, 1)

# --- preparing data for training---
X = pd.DataFrame(data.sentences)
y = pd.DataFrame(data.labels)

# set aside 20% of train and test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, shuffle=True,
                                                    random_state=8)  # 20% for testing and validation
train_df = pd.DataFrame()
train_df["sentences"] = X_train["sentences"]
train_df["labels"] = y_train["labels"]


# --- Fine-tuning ---
# Instantiate Ernie based on the given (str) model name from the CLI.
# NB: Assumes that the number of labels is 2. This can be changed
# with 'labels_no' arg in SentenceClassifier.
#
# Arguments:
# 1 - dataset name (one of: average, majority, intensified)
# 2 - model name e.g. AlbertBaseCased2
# 3 - number of epochs
#
# Example execution: python main.py AlbertBaseCased2.

model_name = sys.argv[2]  # e.g. AlbertBaseCased2
epochs = int(sys.argv[3])  # in range 0..10
classifier = SentenceClassifier(model_name=getattr(Models, model_name))

classifier.load_dataset(train_df, validation_split=0.2)
classifier.fine_tune(
    epochs=epochs,
    learning_rate=2e-5,
    training_batch_size=32,
    validation_batch_size=64
)

# --- Save the model ---
filename = './model_{}_{}_{}'.format(dataset, model_name, epochs)
print("saving as ", filename)
classifier.dump(filename)
print("done")

