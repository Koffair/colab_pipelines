import os

import pandas as pd
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder
from nltk.tokenize import word_tokenize

###############################################################################################
#####                                     data                                            #####
###############################################################################################
clip_prefix = "data/raw/cv-corpus-13.0-2023-03-09/hu/clips"
df = pd.read_csv("data/raw/cv-corpus-13.0-2023-03-09/hu/validated.tsv", sep="\t")
df = df[df["down_votes"] == 0]
clips = df["path"]
clips = [e for e in clips if os.path.isfile(os.path.join(clip_prefix, e)) and os.stat(os.path.join(clip_prefix, e)).st_size != 0]
df = df[df["path"].isin(clips)]
df = df.sample(200)
df["path"] = [os.path.join(clip_prefix, f) for f in df["path"]]


def clean_sentence(sentence):
    wds = word_tokenize(sentence)
    return " ".join([wd.lower() for wd in wds if wd.isalnum()])


df["sentence"] = [clean_sentence(e) for e in df["sentence"]]
print(df.shape)

###############################################################################################
#####                                 load models                                         #####
###############################################################################################
model_path = "models/hs"
lm_path = "models/lms/hu_kenlm.binary"
unigrams_path = "models/lms/unigrams.txt"

references = [{"path": e[0], "transcription": e[1]} for e in zip(df["path"], df["sentence"])]

model = SpeechRecognitionModel(model_path)
decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path)

###############################################################################################
#####                                evaluation                                           #####
###############################################################################################
evaluation = model.evaluate(references, decoder=decoder)

print(evaluation)
###############################################################################################
#####                              transcription                                          #####
###############################################################################################
transcriptions = model.transcribe([k["path"] for k in references], decoder=decoder)
transcriptions = [e["transcription"] for e in transcriptions]
sentences = [e["transcription"] for e in references]

df2 = pd.DataFrame(list(zip(transcriptions, sentences)), columns=["Transcript", "Original"])