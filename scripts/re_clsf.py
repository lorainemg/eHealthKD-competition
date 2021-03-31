from anntools import Collection, Keyphrase, Relation

class REClassifier:
    "Classifier for the relation extraction task"
    def __init__(self):
        pass
   
    def fit(self, collection: Collection):
        keyphrases = self._get_keyphrases(collection)
        # train with the gold keyphrases


    def _get_keyphrases(self, collection):
        "Gets gold keyphrases"
        keyphrases = {}
        for sentence in collection.sentences:
            for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                keyphrases[text] = keyphrase.label
        return keyphrases

    def run(self, collection: Collection):
        collection = collection.clone()
        # returns a collection with everything annotated
        return collection