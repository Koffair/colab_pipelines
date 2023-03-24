import html
import re

import pandas as pd

df = pd.read_csv("data/raw/nyest/contents.csv", sep=";")
df.fillna("", inplace=True)

CLEANR = re.compile("<.*?>")
CDATA = re.compile("\/\/\s&lt;!\[CDATA\[\n.*\n\/\/\s*\]\]&gt;")


def clean_txt(txt):
    """Postprocess txt, removes unescaped html entities"""
    txt = txt.replace("&amp;gt;", " ").replace("&amp;nbsp;", " ").replace("&quot;", " ")
    txt = txt.replace("&#x27", " ").replace("::adbox::7::", "").replace("&amp;lt;", " ")
    txt = txt.replace("&amp;amp;", " ")
    return txt


def cleanhtml(raw_html):
    """Clean raw html page"""
    cleaned_txt = clean_txt(html.escape(re.sub(CLEANR, " ", raw_html)))
    return re.sub(CDATA, " ", cleaned_txt)


with open("data/interim/nyest.txt", "w") as outfile:
    for _, row in df.iterrows():
        txt = []
        if row[0]:
            txt.append(cleanhtml(row[0]))
        if row[1]:
            txt.append(cleanhtml(row[1]))
        if row[2]:
            txt.append(cleanhtml(row[2]))
        if row[3]:
            txt.append(cleanhtml(row[3]))
        if row[4]:
            txt.append(cleanhtml(row[4]))
        outfile.write(" ".join(txt) + "\n")

print("Done")
