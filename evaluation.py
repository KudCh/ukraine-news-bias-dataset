import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ernie import SentenceClassifier, Models
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

dataset = "intensified"
model_name = None

data = pd.read_csv("subjectivity_{}.csv".format(dataset))

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
classifier = SentenceClassifier(model_path='./model_{}_{}'.format(dataset, model_name))
print("model loaded")

#--- Predicting ---
X_test_list = X_test["sentences"].values.tolist()
probabilities = [p for p in classifier.predict(X_test_list)]
array_probs = np.array(probabilities)


y_pred = np.argmax(array_probs, axis=1)

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

#Bert Based Cased intensified vote
#41/41 [==============================] - 1258s 30s/step - loss: 0.4889 - accuracy: 0.8155 - val_loss: 0.4087 - val_accuracy: 0.8469
#41/41 [==============================] - 840s 21s/step - loss: 0.4283 - accuracy: 0.8232 - val_loss: 0.3831 - val_accuracy: 0.8469
#41/41 [==============================] - 4593s 114s/step - loss: 0.3486 - accuracy: 0.8331 - val_loss: 0.3999 - val_accuracy: 0.8219
#41/41 [==============================] - 695s 17s/step - loss: 0.2434 - accuracy: 0.8864 - val_loss: 0.5784 - val_accuracy: 0.8375

#try other models - Roberta, bert large, etc.
#tabke of results with different models
#graph od different number of epochs (1 to 10??)
# make a CLI with model arguments