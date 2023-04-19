alphabet = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"


def hungarian_word(strng):
    for ch in strng:
        if ch not in alphabet:
            return False
    return True


with open("data/interim/merged_corpus_cleaned.txt", "w") as outfile:
    with open("data/interim/merged_corpus.txt", "r") as infile:
        for l in infile:
            l = l.split()
            l = [wd for wd in l if wd.isalpha() and hungarian_word(wd)]
            l = " ".join(l)
            if l:
                outfile.write(l + "\n")
