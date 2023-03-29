import os

import pandas as pd
from huggingsound import (
    ModelArguments,
    SpeechRecognitionModel,
    TokenSet,
    TrainingArguments,
)
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

########################################################################################################
#####                               Getting some data                                              #####
########################################################################################################
clip_prefix = "data/raw/cv-corpus-13.0-2023-03-09/hu/clips"

df = pd.read_csv("data/raw/cv-corpus-13.0-2023-03-09/hu/validated.tsv", sep="\t")
df = df[df["down_votes"] == 0]  # use only validated data without down_votes
clips = df["path"]
clips = [e for e in clips if os.path.isfile(os.path.join(clip_prefix, e)) and os.stat(os.path.join(clip_prefix, e)).st_size != 0]
df = df[df["path"].isin(clips)]

print(df.shape)  # 28891
train, test = train_test_split(df, test_size=0.2)

trainx = zip(train["path"], train["sentence"])
testx = zip(test["path"], test["sentence"])


def clean_sentence(sentence):
    wds = word_tokenize(sentence)
    return " ".join([wd.lower() for wd in wds if wd.isalnum()])


train_data = [
    {"path": os.path.join(clip_prefix, e[0]), "transcription": clean_sentence(e[1])}
    for e in trainx
]
eval_data = [
    {"path": os.path.join(clip_prefix, e[0]), "transcription": clean_sentence(e[1])}
    for e in testx
]

########################################################################################################
#####                                  Model setup                                                 #####
########################################################################################################
model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
output_dir = "models/"

tokens = [
    "a",
    "á",
    "b",
    "c",
    "d",
    "e",
    "é",
    "f",
    "g",
    "h",
    "i",
    "í",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ó",
    "ö",
    "ő",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "ú",
    "ü",
    "ű" "v",
    "w",
    "x",
    "y",
    "z",
]
token_set = TokenSet(tokens)

########################################################################################################
#####                                    Fine-tune model                                           #####
########################################################################################################
model.finetune(
    output_dir,
    train_data=train_data,
    eval_data=eval_data,
    token_set=token_set,
)
