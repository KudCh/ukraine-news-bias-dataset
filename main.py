import os
from ernie import SentenceClassifier, Models

models = [
    "BertBaseCased",  # 0
    "BertLargeCased",  # 1
    "AlbertBaseCased",  # 2
    "AlbertLargeCased",  # 3
    "AlbertXLargeCased",  # 4
    "AlbertXXLargeCased",  # 5

    "DistilBertBaseMultilingualCased",  # 6
    "RobertaBaseCased",  # 7
    "RobertaLargeCased",  # 8
    "XLNetBaseCased",  # 9
    "XLNetLargeCased"  # 10
]

#  Bert base, Bert Large, Albert Base, Roberta Base, XLNet base
for model in ["BertLargeCased"]:
    classifier = SentenceClassifier(model_name=getattr(Models, model))
    classifier.dump(model)
    for i in range(5):
        for dataset in ["majority", "intensified"]:
            os.system("python model_preps.py {dataset} {model} {epochs}".format(dataset=dataset,
                                                                            model=model,
                                                                            epochs=i)
                  )
