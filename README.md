# Natural Language Processing with Latent Dirichlet Allocation (LDA)

## Overview

This project explores the application of Latent Dirichlet Allocation (LDA) in Natural Language Processing (NLP) using Python. 
LDA is a probabilistic model commonly used for topic modeling, which helps uncover hidden thematic structures within a collection 
of documents. By identifying topics from text data, LDA enables automatic categorization and summarization, making it a powerful 
tool in textual analysis and information retrieval.

## Key Components

### 1. Libraries Used

- **Gensim**: A Python library for topic modeling, text similarity, and document indexing.
- **NLTK (Natural Language Toolkit)**: A suite of libraries and programs for natural language processing tasks such as tokenization,
                                       stemming, tagging, parsing, and semantic reasoning.

### 2. Preprocessing

Before applying LDA, the text undergoes preprocessing to enhance the quality of topic extraction:

- **Tokenization**: Splits text into individual words or tokens.
- **Stopword Removal**: Eliminates common words (e.g., "and", "the") that do not contribute to topic identification.
- **Lemmatization**: Reduces words to their base or root form to normalize variations (e.g., "running" to "run").

### 3. Building the LDA Model

- **Dictionary Creation**: Constructs a dictionary mapping of all unique words found in the corpus.
- **Corpus Creation**: Generates a numerical representation (bag-of-words) of each document using the dictionary.
- **LDA Model Training**: Applies the LDA algorithm to discover a specified number of topics within the corpus. Parameters such as the
                          number of topics (`num_topics`) and the number of passes (`passes`) through the corpus can be adjusted to
                          optimize topic coherence.

### 4. Evaluation

- **Coherence Score**: Measures the semantic coherence of topics. A higher coherence score indicates more interpretable and distinct topics.

## Example Usage

### Sample Code Snippet

```python
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
import string

# Sample documents
documents = [
    "Artificial intelligence and robotics are leading the next wave of digital transformation.",
    "Global markets are increasingly volatile, affecting international trade and investment strategies.",
    "Advancements in biotechnology are making personalized medicine more accessible than ever.",
    "Climate change is the defining issue of our time, affecting global weather patterns and ecosystems.",
    "Quantum computing could revolutionize data processing by significantly speeding up problem-solving capabilities."
]

# Preprocessing
download('stopwords')
download('punkt')
download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

processed_docs = [preprocess_text(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Build LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10)

# Print the topics
for topic_id, topic in lda_model.print_topics():
    print(f"Topic {topic_id}: {topic}")

# Compute coherence score
coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score}")
```

## Conclusion

This project demonstrates the effective use of LDA for topic modeling in NLP. By analyzing a diverse set of textual data, 
the LDA model identifies coherent topics that represent the underlying themes within the documents. The combination of Gensim 
and NLTK libraries provides robust tools for preprocessing text and extracting meaningful insights from unstructured data.

## Future Improvements

- **Parameter Optimization**: Experiment with different values for `num_topics` and `passes` to enhance topic quality.
- **Advanced Preprocessing**: Implement advanced techniques like Named Entity Recognition (NER) for better topic identification.
- **Integration**: Extend the project to incorporate other NLP tasks such as sentiment analysis or text summarization using the identified topics.

## Author

Natnael Haile
