import math, os, pickle, re
from typing import Tuple, List, Dict




class BayesClassifier:
    """A simple BayesClassifier implementation


    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """


    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"


        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()


    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier"""


        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")


        for index, filename in enumerate(files, 1):  # type: ignore
            print(f"Training on file {index} of {len(files)}: {filename}")


            path = os.path.join(self.training_data_directory, filename)
            text = self.load_file(path)
            tokens = self.tokenize(text)


            # determine class
            if filename.startswith(self.neg_file_prefix):
                self.update_dict(tokens, self.neg_freqs)
            elif filename.startswith(self.pos_file_prefix):
                self.update_dict(tokens, self.pos_freqs)
            else:
                continue  # ignore neutral files


        # save dictionaries
        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)


    def classify(self, text: str) -> str:
        """Classifies given text as positive or negative"""


        tokens = self.tokenize(text)


        pos_prob = 0.0
        neg_prob = 0.0


        pos_total = sum(self.pos_freqs.values())
        neg_total = sum(self.neg_freqs.values())


        for tok in tokens:
            pos_count = self.pos_freqs.get(tok, 0)
            neg_count = self.neg_freqs.get(tok, 0)


            # add-one smoothing
            pos_p = (pos_count + 1) / (pos_total + 1)
            neg_p = (neg_count + 1) / (neg_total + 1)


            pos_prob += math.log(pos_p)
            neg_prob += math.log(neg_p)


        if pos_prob > neg_prob:
            return "positive"
        else:
            return "negative"


    def load_file(self, filepath: str) -> str:
        """Loads text of given file"""
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()


    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name"""
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)


    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file"""
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()


    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order"""
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))


        if token != "":
            tokens.append(token.lower())
        return tokens


    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list"""
        for w in words:
            if w in freqs:
                freqs[w] += 1
            else:
                freqs[w] = 1




if __name__ == "__main__":
    b = BayesClassifier()


    # Test some sentences
    print(b.classify("I love this movie"))
    print(b.classify("this is terrible"))
    print(b.classify("Great movie"))
    print(b.classify("I love python!")) # should print positive
    print(b.classify("I hate python!")) # should print negative
