# colab_pipelines
**WARNING**: You can run the notebooks on Colab, but
some of the notebooks requires subscription. Have a look
at the src folder and run most of the jobs on a local machine.
More details below.

# TODO:
+ write readme
+ link data folders
+ etc

# Data
Team members can access the data via Google Drive. Click [here](https://drive.google.com/drive/folders/1R4AFvTP91Lb5xxp5u1y6HWkA5xhgm7Ai?usp=share_link)
+ SST was trained on the latest [Hungarian Common Voice corpus](https://commonvoice.mozilla.org/)
+ For output correction we trained a floret model using the Hungarian sub-corpus of [OSCAR 2019](https://oscar-project.org/post/oscar-2019/) and 
all the articles from [nyest.hu](https://www.nyest.hu/)

# Running the scripts
Scripts in the ```src``` folder are almost identical to the scripts in ```notebooks```.
The main difference between the two version is that paths in the scripts are relative paths
while the notebooks contains absolute paths on Google Drive.

The other minor difference is that bash commands like unzipping and concatenating files are not
presented in the scripts. Running command line utilities like ```floret``` and ```KenLM``` are
shown as special cells in the notebooks.

# Build language models
Our merged corpus (nyest + OSCAR19) contains 4466526 lines. 
```bash
wc -l data/interim/merged_corpus.txt
```
On a single laptop/PC it
takes ages to train a language model. You can take a sample from the corpus using
the following command
```bash
shuf -n 1000 data/interim/merged_corpus.txt > data/interim/sample1000.txt
```

## Training a floret model
```bash
../../opt/floret/floret cbow -dim 300 -minn 3 -maxn 6 -mode floret -hashCount 4 -bucket 50000 -input data/interim/merged_corpus.txt -output models/lms/hufloret_
```

## Training a KenLM language model
First things first, we have to clean up our corpus, so it will
contain only characters of the Hungarian alphabet.
Run `src/data_tasks/preprocess_merged.py`. We need a vocabulary
file too, let's generate it by running `**src/data_tasks/get_unigrams.py`

Now, let's build a trigram model
```bash
../../opt/kenlm/build/bin/lmplz -o 4 < data/interim/merged_corpus_cleaned.txt > models/lms/hu_kenlm.arpa
```
You have to modify the resulting language model since it doesn't contain a few types required by HF.
Run `src/data_tasks/post_process_kenml.py`
Let's make a smaller, binary version of the LM.
```bash
../../opt/kenlm/build/bin/build_binary models/lms/hu_kenlm_corrected.arpa models/lms/hu_kenlm.binary
```