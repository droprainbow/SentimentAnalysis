import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class DataCleaner:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        self.stop.update(punctuation)

    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def rm_between_square_brackets(self, text):
        return re.sub('\[[^]]*\]', "", text)

    def rm_url(self, text):
        return re.sub(r'http\S+', '', text)

    def rm_stopwords(self, text):
        res = []
        for i in text.split():
            if i.strip().lower() not in self.stop:
                res.append(i.strip())

        return ' '.join(res)

    def denoise_text(self, text):
        text = self.strip_html(text)
        text = self.rm_between_square_brackets(text)
        text = self.rm_url(text)
        text = self.rm_stopwords(text)

        return text
