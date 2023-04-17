import icu

vocab = set()


with open("data/interim/merged_corpus_cleaned.txt", "r") as infile:
    for l in infile:
        wds = l.strip().split()
        vocab.update(wds)

collator = icu.Collator.createInstance(icu.Locale("hu_HU.UTF-8"))
unigrams = sorted(vocab, key=collator.getSortKey)

with open("models/lms/unigrams.txt", "w") as outfile:
    for wd in unigrams:
        outfile.write(wd + "\n")
