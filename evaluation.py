import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ernie import SentenceClassifier, Models
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

model = "intensified"

data = pd.read_csv("subjectivity_{}.csv".format(model))

#2 classes:
# < 1, >= 1
# 0: objective, 1: subjective
data["labels"] = np.where(data["scores"] < 1.0, 0, 1)

#--- preparing data for training---
X = pd.DataFrame(data.sentences)
y = pd.DataFrame(data.labels)

# set aside 20% of train and test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, shuffle = True, random_state = 8) # 20% for testing and validation


#--- Load the model ---
print("model loading")
classifier = SentenceClassifier(model_path='./model_{}1'.format(model))
print("model loaded")

#--- Predicting ---
X_test_list = X_test["sentences"].values.tolist()
probabilities = [p for p in classifier.predict(X_test_list)]
array_probs = np.array(probabilities)

y_pred = []

for array in array_probs:
    y_pred.append(array.argmax())

#--- Evaluating ---

accuracy = balanced_accuracy_score(y_test, y_pred)
print(accuracy)
#average
#0.6623142129471243
#majority
#0.55
#intensified
#0.5826527996484419
roc_auc = roc_auc_score(y_test, y_pred)
print(roc_auc)
#average
#0.6623142129471243
#majority
#0.55
#intensified
#0.5826527996484417
