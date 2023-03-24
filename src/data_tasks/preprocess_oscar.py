import concurrent.futures
import gzip
import os

from blingfire import text_to_words

data_root = "data/raw/OSCAR2019_hu"
text_files = [
    e for e in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, e))
]

with open("data/interim/oscar.txt", "w") as outfile:
    for text_file in text_files:
        print(text_file)
        with gzip.open(os.path.join(data_root, text_file), "rt") as infile:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                res = {executor.submit(text_to_words, line) for line in infile}
                for future in concurrent.futures.as_completed(res):
                    data = future.result()
                    wds = data.split()
                    wds = [wd.lower() for wd in wds if wd.isalnum()]
                    wds = " ".join(wds)
                    outfile.write(wds + "\n")
print("Done")
