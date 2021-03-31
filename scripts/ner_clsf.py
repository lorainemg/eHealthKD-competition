from anntools import Collection, Keyphrase, Relation

class NERClassifier:
    "Classifier for the name entity resolution task"
    def __init__(self):
        pass
   
    def fit(self, collection: Collection):
        pass

    def run(self, collection: Collection):
        collection = collection.clone()
        # returns a collection with everything annotated
        return collection