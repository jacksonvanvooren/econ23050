import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.corpora import WikiCorpus
from gensim.utils import simple_preprocess

### Set up Word2Vec Model
# Path to the Wikipedia dumps
dump_path_1 = '/Users/jacksonvanvooren/Downloads/enwiki-20241001-pages-articles-multistream1.xml-p1p41242.bz2'
dump_path_2 = '/Users/jacksonvanvooren/Downloads/enwiki-20241001-pages-articles-multistream2.xml-p41243p151573.bz2'

wiki_corpus_1 = WikiCorpus(dump_path_1, dictionary={})
wiki_corpus_2 = WikiCorpus(dump_path_2, dictionary={})

# Keywords
keywords = ['apple', 'blackberry']

tokenized_corpus = []

# Process based on the keywords
for i, text in enumerate(wiki_corpus_1.get_texts()):
    article_text = " ".join(text)
    if any(keyword.lower() in article_text.lower() for keyword in keywords):
        tokenized_corpus.append(simple_preprocess(article_text))

for i, text in enumerate(wiki_corpus_2.get_texts()):
    article_text = " ".join(text)
    if any(keyword.lower() in article_text.lower() for keyword in keywords):
        tokenized_corpus.append(simple_preprocess(article_text))

model = Word2Vec(
    sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, sg=1
)
model.save("word2vec_filtered_model")
loaded_model = Word2Vec.load("word2vec_filtered_model")


### Query and print model results
# Get the top 10 most similar words to 'apple' and 'blackberry'
apple_similar_words = loaded_model.wv.most_similar("apple", topn=10)
blackberry_similar_words = loaded_model.wv.most_similar("blackberry", topn=10)

print("Most similar words to 'Apple':")
for word, similarity in apple_similar_words:
    print(f"{word}: {similarity}")

print("\nMost similar words to 'BlackBerry':")
for word, similarity in blackberry_similar_words:
    print(f"{word}: {similarity}")

# Calculate similarity scores
innovation_words = ["innovation", "creativity", "design", "progress"]
apple_similarities = {}
blackberry_similarities = {}

for word in innovation_words:
    apple_similarities[word] = loaded_model.wv.similarity("apple", word)

for word in innovation_words:
    blackberry_similarities[word] = loaded_model.wv.similarity("blackberry", word)

apple_sim_scores = [apple_similarities[word] for word in innovation_words]
blackberry_sim_scores = [blackberry_similarities[word] for word in innovation_words]


### Plotting
x_labels = innovation_words
x_pos = np.arange(len(x_labels))
plt.figure(figsize=(12, 6))
plt.bar(x_pos - 0.2, apple_sim_scores, width=0.4, color='blue', label="Apple", alpha=0.6)
plt.bar(x_pos + 0.2, blackberry_sim_scores, width=0.4, color='red', label="BlackBerry", alpha=0.6)

plt.xticks(x_pos, x_labels, rotation=45)
plt.xlabel("Innovation-related Words")
plt.ylabel("Similarity Score")
plt.title("Comparison of Similarity: Apple vs BlackBerry with Innovation-related Words")
plt.legend()
plt.tight_layout()
plt.show
