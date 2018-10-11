import os

def preprocess():
    sentences_path = os.path.join("datasets", "stanford_sentiment", "stanfordSentimentTreebank", "datasetSentences.txt")
    newlines = []
    with open(sentences_path, "r") as f:
        lines = f.readlines()
        for line in lines:
                splitted = line.strip().split()
                sentence_splitted = [w.lower().decode('utf-8').encode('latin1') for w in splitted]
                fixed = " ".join(sentence_splitted)
                print fixed
                newlines.append(fixed + "\n")

    with open(sentences_path, "w") as fo:
        fo.writelines(newlines)

if __name__ == "__main__":
    preprocess()
