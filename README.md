# 🎥 IMDB Movie Review Sentiment Analysis 📊

This project is a sentiment analysis model that classifies IMDB movie reviews as either **positive** or **negative** using an **Artificial Neural Network (ANN)**. The model is trained on the IMDB dataset using the **SimpleRNN** architecture in TensorFlow/Keras. The web application is built using **Streamlit**, allowing users to input their movie reviews and get real-time sentiment predictions.

## 🚀 Streamlit Web Application

This project comes with an interactive Streamlit web application, where users can enter a movie review, and the model predicts its sentiment.

<p align="center">
  <a href="https://your-streamlit-app-link">
    <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="250" alt="Streamlit Logo">
  </a>
</p>

### Key Features:
- **Sentiment Analysis** on IMDB movie reviews
- **SimpleRNN-based model** to capture the sequential nature of text data
- **Interactive UI** to classify reviews as either positive or negative in real-time
- **Streamlit App** for ease of use

## 📚 Table of Contents
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Preprocessing](#-preprocessing)
- [Prediction Pipeline](#-prediction-pipeline)
- [Usage](#-usage)

## 📂 Project Structure

```
📂 Movie_Review_End_to_End_NLP/
├── 📁 Models/
│   ├── model.h5         
├── 📂 Notebooks/
│   └── experiments.ipynb
│   └── embeddings.ipynb
├── app.py                   # Streamlit app source code
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
```



## 💽 Dataset

The model is trained on the **IMDB Movie Reviews** dataset, which contains 50,000 reviews labeled as positive or negative. This dataset is available directly through the **Keras Datasets API**.

### Dataset Highlights:
- **Number of Reviews**: 50,000
- **Labels**: Positive (1) and Negative (0)
- **Reviews**: Pre-tokenized and padded to a fixed length of 500 words

## 🤖 Model Architecture

The sentiment classification model is built using a **SimpleRNN** architecture. The key layers are:

- **Embedding Layer**: Transforms words into dense vectors of fixed size.
- **SimpleRNN Layer**: Captures the sequential dependencies in the text.
- **Dense Layer**: A fully connected layer with a sigmoid activation function to output probabilities.

### Model Details:
- **Activation Functions**: ReLU for hidden layers and Sigmoid for the output layer.
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam.

## 🧹 Preprocessing

The text preprocessing involves:
1. **Tokenization**: Mapping words to integer indices using Keras's built-in IMDB dataset word index.
2. **Padding**: Padding sequences to a fixed length of 500 words to ensure uniform input to the model.
3. **Encoding**: Encoding input reviews based on their corresponding indices from the IMDB dataset's word index.

## 📝 Prediction Pipeline

The model uses a review input from the user, which undergoes the following steps:
1. **Text Preprocessing**: Converts the review into a padded sequence of word indices.
2. **Model Prediction**: The preprocessed input is passed to the trained SimpleRNN model.
3. **Sentiment Classification**: The model predicts whether the review is **positive** or **negative** based on a threshold of 0.5.

## 💻 Usage

### Running the App Locally

To run the Streamlit app locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Movie_Review_Sentiment_Analysis.git
    cd Movie_Review_Sentiment_Analysis
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:
    ```bash
    streamlit run app.py
    ```
