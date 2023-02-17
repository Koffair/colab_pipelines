import os

from nltk.tokenize import sent_tokenize, word_tokenize

data_root = "data/raw/oscar19"
text_files = [
    e for e in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, e))
]

with open("data/interim/oscar.txt", "w") as outfile:
    for text_file in text_files:
        with open(os.path.join(data_root, text_file), "r") as infile:
            for line in infile:
                sentences = sent_tokenize(line)
                for sentence in sentences:
                    if sentence:
                        words = word_tokenize(sentence)
                        words = [word for word in words if word.isalnum()]
                        if words:
                            s = " ".join(words)
                            outfile.write(s + "\n")
print("Done")
