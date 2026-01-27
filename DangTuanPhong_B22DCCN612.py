import nltk
from collections import Counter

# Download cần thiết (chỉ cần chạy 1 lần)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Sentence
sentence = "Natural language processing enables computers."

# Tokenize
tokens = nltk.word_tokenize(sentence)

# POS tagging (Penn Treebank)
pos_tags = nltk.pos_tag(tokens)

# Print (word, POS) pairs
print("Word - POS tag pairs:")
for word, tag in pos_tags:
    print(f"{word} - {tag}")

# Count POS tag frequency
pos_counts = Counter(tag for word, tag in pos_tags)

print("\nPOS tag frequencies:")
for tag, count in pos_counts.items():
    print(f"{tag}: {count}")

# Top 3 most frequent POS tags
top_3 = pos_counts.most_common(3)

print("\nTop 3 most frequent POS tags:")
for tag, count in top_3:
    print(f"{tag}: {count}")
