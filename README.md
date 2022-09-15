# UH-MMM submission for the eHealth-KD Challenge

This is the source code of our team final submission.

`2021` contains the data available for the competition. This includes training, development and test data.

`scripts` contains the source code of the different models implemented. In `ner_clsf.py` the NER model is implemented and `re_clsf.py` contains the RE model implementation. `classifier.py` wraps the name entity classifier and the relation extractor classifier, the models are tested in the different scenarios and collections.

The FastText Spanish Medical Embeddings are not included for its size. They can be downloaded in https://zenodo.org/record/3744326. For the system to work, the Scielo+Wiki SkipGram Uncased should be unzipped in `scripts` (i.e, the bin file)

The documentation paper can be found in [paper](https://github.com/lorainemg/eHealthKD-competition/blob/main/docs/ehealth_paper4.pdf).

