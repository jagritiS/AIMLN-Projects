import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import logging
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Enable logging for gensim to monitor the training process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_text(text):
    """
    Preprocess the input text: remove non-alphabetic characters, tokenize, and remove stopwords.
    """
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def topic_modeling(corpus):
    """
    Apply LDA topic modeling on the corpus and return the topics.
    """
    # Preprocess the corpus
    processed_corpus = [preprocess_text(doc) for doc in corpus]

    # Create a dictionary from the processed corpus
    dictionary = corpora.Dictionary(processed_corpus)

    # Create a bag-of-words (BoW) representation of the corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_corpus]
    
    # Train an LDA model
    lda_model = LdaModel(bow_corpus, num_topics=3, id2word=dictionary, passes=15)
    
    return lda_model, bow_corpus, dictionary

def print_topics(lda_model):
    """
    Print the topics and their top words.
    """
    print("\nTopics found by LDA:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic #{idx + 1}: {topic}")

def visualize_topics(lda_model, bow_corpus, dictionary):
    """
    Visualize the LDA topics using pyLDAvis.
    """
    # Prepare the visualization
    vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
    
    # Save the visualization as an HTML file
    pyLDAvis.save_html(vis, 'lda_visualization.html')

    # Optionally, open the visualization HTML file in a browser (Windows)
    if os.name == 'nt':  # For Windows, use os.system to open the file
        os.system('start lda_visualization.html')
    elif os.name == 'posix':  # For macOS/Linux
        os.system('open lda_visualization.html')
    
    print("\nTopic visualization saved as 'lda_visualization.html'")

def main():
    print("Topic Modeling using LDA (Latent Dirichlet Allocation)\n")
    
    # Sample corpus (list of text documents)
    corpus = [
        "I love programming in Python. Python is a versatile language.",
        "Data science and machine learning are fascinating fields.",
        "I enjoy building machine learning models using scikit-learn.",
        "Natural language processing helps in understanding human language.",
        "The Python ecosystem has great tools for machine learning.",
        "Deep learning models have shown excellent performance in NLP tasks."
    ]
    
    # Perform topic modeling
    lda_model, bow_corpus, dictionary = topic_modeling(corpus)
    
    # Print the topics discovered by LDA
    print_topics(lda_model)
    
    # Visualize the topics using pyLDAvis
    visualize_topics(lda_model, bow_corpus, dictionary)

if __name__ == "__main__":
    main()
