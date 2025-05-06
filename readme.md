# Clinical Disease Prediction System

This project leverages multiple NLP models—including a fine-tuned BERT, LSTM with GloVe, LSTM with BioBERT, and a hybrid LSTM (GloVe + BioBERT)—to predict diseases based on user-input symptoms. The system features a command-line interface (CLI) that evaluates an input sentence across all models and selects the prediction with the highest confidence.

## Table of Contents
- [Google Drive](#google-drive)
- [Installation](#installation)
- [Dataset and Embeddings](#dataset-and-embeddings)
- [Running the Code](#running-the-code)
  - [Training and Evaluation](#training-and-evaluation)
  - [Running Predictions via the CLI](#running-predictions-via-the-cli)
- [Additional Notes](#additional-notes)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

## Google Drive
All model implementations, datasets, and results are available via our shared drive:

- **Drive Link**: [Clinical Disease Prediction Models](https://drive.google.com/drive/folders/1e0q8ZVFNUQZnO9O--X3TxsFzmmR-iRmq?usp=sharing)

### Repository Structure

Each model folder in the drive contains:

- **Python Code**: Implementation scripts for the specific model variant
- **Pretrained Model(s)**: Saved model checkpoints ready for inference
- **Results**: Text file documenting performance metrics and evaluation results

The drive also includes the complete Kaggle dataset used for training and evaluation.


## Installation

1. **Install Python 3.13 (if not already installed):**

## Dataset and Embeddings

### Download the Dataset:

The system uses the "kuc-hackathon-winter-2018" dataset from Kaggle. The code automatically downloads it if the dataset directory (./kuc-hackathon-winter-2018) is not found.

Note: Ensure your Kaggle API credentials are configured if required.

### Download GloVe Embeddings:

The project expects the GloVe embeddings file `glove.6B.300d.txt` to be available. If running locally, download and extract the embeddings:

Using CLI (if available):
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Ensure that `glove.6B.300d.txt` is placed in the expected directory (or adjust the path in the code accordingly).

## Running the Code

### Training and Evaluation for individual model

We have seperate scripts that can be run with command-line flags to train or evaluate the models.

#### Train the Model:

```bash
python 
```

This command will:
- Download the dataset if necessary.
- Build the vocabulary from the clinical data.
- Load embeddings (GloVe and optionally supplement with BioBERT).
- Train the model using a combined training-evaluation-prediction workflow.
- Save the best model checkpoint (e.g., at `./best_lstm_model.pt`).

#### Evaluate the Model:

```bash
python 
```

This command loads the saved model and evaluates its performance on the test dataset.

### Running Predictions via the CLI

To run a prediction on a single input sentence, use the `predict_disease.py` script. For example:

```bash
python predict_disease.py "I have dry, itchy, red patches on my skin that are scaly and sometimes develop small blisters that ooze."
```

The CLI will:
- Run predictions using all model variants (BERT, LSTM with BioBERT, LSTM with GloVe, and LSTM with GloVe+BioBERT).
- Display the top 5 conditions with confidence scores for each model.
- Output the final prediction from the model with the highest confidence.

## Additional Notes

### Logging and Output:
The CLI prints detailed output for each model variant, including prediction confidence scores. It then selects the model with the highest confidence for the final prediction.

### Citing Pre-trained Models:
If you use the fine-tuned BERT model in your research, please consider citing the following:

Bhargava et al. (2021)
```
@misc{bhargava2021generalization,
      title={Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics}, 
      author={Prajjwal Bhargava and Aleksandr Drozd and Anna Rogers},
      year={2021},
      eprint={2110.01518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Turc et al. (2019)
```
@article{DBLP:journals/corr/abs-1908-08962,
  author    = {Iulia Turc and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {Well-Read Students Learn Better: The Impact of Student Initialization
               on Knowledge Distillation},
  journal   = {CoRR},
  volume    = {abs/1908.08962},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.08962},
  eprinttype = {arXiv},
  eprint    = {1908.08962}
}
```

## Troubleshooting

### Environment Issues:
If you encounter version incompatibilities (e.g., with NumPy or gensim), try downgrading or upgrading the packages as necessary and ensure you restart your virtual environment.

## Conclusion
This project represents a robust approach to clinical disease prediction by leveraging diverse deep learning models and embeddings. Our CLI, which selects the best prediction based on confidence scores from multiple models, not only provides actionable diagnostic insights but also helps analyze model performance across different types of clinical input. Follow the above instructions to set up, train, evaluate, and deploy the system.