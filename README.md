# Natural_Language_Processing
🤖💻This repository showcases a comprehensive Natural Language Processing (NLP) pipeline implemented in Python using Jupyter notebooks. The pipeline deploys various machine learning techniques to classify labeled dataset. The pipeline employs comparisons of the dataset using Recurrent Neural Network (RNN) and RandomForest Classifier algorithms.

## KEY FEATURES:

1. Text Processing: The NLP pipeline encompasses essential text processing techniques such as normalization, tokenization, and stop words removal. These steps ensure standardized and clean input data for subsequent analysis.
2. Feature Extraction: The pipeline utilizes advanced feature extraction methods to represent sentences effectively. It leverages popular approaches such as doc2vec and word2vec to transform text into numerical vectors, capturing semantic and contextual information.
3. Modeling: The NLP pipeline employs machine learning models to train and classify text data. It includes comparisons of the dataset using RNN and RandomForest Classifier algorithms, enabling evaluation and selection of the most suitable model for text classification tasks.
4. Python Libraries: The repository extensively utilizes various Python libraries for NLP, including but not limited to:
a. NLTK (Natural Language Toolkit) for text processing and tokenization.
b. Gensim for implementing doc2vec and word2vec models.
c. Scikit-learn for feature extraction, modeling, and evaluation.
d. Keras for building and training RNN models.
e. RandomForest Classifier from the scikit-learn library for comparative analysis.
5. Evaluation Metrics: The accuracy of each feature representation and model is measured to assess the performance of the NLP pipeline. The achieved accuracies of 0.902 for Document vectors, 0.973 for tf-idf vectors, and 0.877 for word vectors demonstrate the effectiveness of the implemented approaches.
6. Dataset: The repository includes a labeled dataset containing sentences classified as ham or spam. This dataset serves as the foundation for training and evaluating the NLP pipeline.

## ACCESS

Project Jupyter’s tools are available for installation via the [Python Package Index](https://pypi.org/), the leading repository of software created for the Python programming language.

Here are instructions with [pip](https://pip.pypa.io/en/stable/), [the recommended installation tool for Python](https://packaging.python.org/en/latest/guides/tool-recommendations/#installation-tool-recommendations). If you require environment management as opposed to just installation, look into [conda](https://docs.conda.io/en/latest/), [mamba](https://mamba.readthedocs.io/en/latest/), and [pipenv](https://pipenv.pypa.io/en/latest/).

Jupyter Notebook:

Install the classic Jupyter Notebook by writing the following commands in the terminal:

pip install notebook

To run the notebook:

jupyter notebook


## SUMMARY

By providing a comprehensive NLP pipeline implemented in Jupyter notebooks, this repository offers valuable insights into text classification tasks. Whether you're a data scientist, researcher, or enthusiast, this repository can serve as a valuable resource for understanding and implementing NLP techniques. Explore the notebooks, experiment with different feature representations, and enhance your knowledge of text classification today!

## LICENSE

[MIT LICENSE](LICENSE)
