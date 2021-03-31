import argparse
import re
from typing import List
import warnings
from pathlib import Path

from anntools import Collection, Keyphrase, Relation

class Classifier:
    "Classifier for the main task"
    def __init__(self):
        self.model1 = None
        self.model2 = None

    # scenarios = {
    #     1: ("scenario1-main", True, True),
    #     2: ("scenario2-taskA", True, False),
    #     3: ("scenario3-taskB", False, True),
    # }

    def fit(self, path: Path):
        collection = Collection().load_dir(path)

        print(f"Loaded {len(collection)} sentences for fitting.")

        self.model = keyphrases, relations = {}, {}

        for sentence in collection.sentences:
            for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                keyphrases[text] = keyphrase.label

        for sentence in collection.sentences:
            for relation in sentence.relations:
                origin = relation.from_phrase
                origin_text = origin.text.lower()
                destination = relation.to_phrase
                destination_text = destination.text.lower()

                relations[
                    origin_text, origin.label, destination_text, destination.label
                ] = relation.label

        print(f"Training completed: Stored {len(keyphrases)} keyphrases and {len(relations)} relation pairs.")


    def eval(self, path: Path, scenarios: List[int], submit: Path):
        for id in scenarios:
            folder, taskA, taskB = self.scenarios[id]

            scenario = path / folder
            print(f"Evaluating on {scenario}.")

            input_data = Collection().load(scenario / "input.txt")
            print(f"Loaded {len(input_data)} input sentences.")
            output_data = self.run(input_data, taskA, taskB)

            print(f"Writing output to {submit / folder}")
            output_data.dump(submit / folder / "output.txt", skip_empty_sentences=False)


    def run(self, collection, taskA, taskB):
        # gold_keyphrases, gold_relations = self.model
        collection = collection.clone()

        if taskA:
            # Call to classifier of task a
            pass
        if taskB:
            # call to classifier of task b
            pass
        return collection


def main():
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--ref", required=True, type=Path, help="Location of the reference data.")
    # parser.add_argument("--eval", required=True, type=Path, help="Location of the evaluation data.")
    # parser.add_argument("--scenarios", type=int, nargs="+", help="Scenarios to run.", default=[1,2,3])
    # parser.add_argument("--submit", required=True, type=Path, help="Location to output the submission.")

    # args = parser.parse_args()

    # clsf = Classifier()
    # clsf.fit(args.ref)
    # clsf.eval(args.eval, args.scenarios, args.submit)
    
    ref_ = Path('2021/ref/training')
    eval_ = Path('2021/eval/develop')
    scenarios = [1, 2, 3]
    submit_ = Path('2021/submissions/classifier/develop/run1')

    clsf = Classifier()
    clsf.fit(ref_)
    clsf.eval(eval_, scenarios, submit_) 

    
if __name__ == "__main__":
    main()
