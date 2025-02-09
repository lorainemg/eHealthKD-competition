# eHealthKD-competition

This repository contains the final submission for the eHealth-KD Challenge by the UH-MMM team. The project focuses on Named Entity Recognition (NER) and Relation Extraction (RE) using machine learning models.

## Project Structure

- **2021**: Contains the data available for the competition, including training, development, and test datasets.
- **scripts**: Contains the source code for the implemented models.
  - `ner_clsf.py`: NER model implementation.
  - `re_clsf.py`: RE model implementation.
  - `classifier.py`: Wraps the NER and RE models for testing in various scenarios.

## Setup Instructions

To use the system:
1. Download the FastText Spanish Medical Embeddings from [Zenodo](https://zenodo.org/record/3744326).
2. Unzip the embeddings file "Scielo+Wiki SkipGram Uncased" into the `scripts` directory.

## License

This project is licensed under the MIT License.

## Citation

If you use this code, please refer to the associated paper for more details.

For further information, refer to the [documentation paper](https://github.com/lorainemg/eHealthKD-competition/blob/main/docs/ehealth_paper4.pdf).


