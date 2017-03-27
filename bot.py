from gensim.models import Word2Vec as Word2Vec
from nltk.corpus import gutenberg as gutenberg
import pickle
import numpy
import logging
import utils
import math

class Bot:
    def __init__(self, lines = 10):
        self.read()
        self.write(lines)

    def read(self):
        try:
            print('trying to remember...')
            self.model = pickle.load(open('model.bin', 'rb'))
            print('remembered previous reading session.')
        except Exception as e:
            print(e)
            print('reading...')
            self.model = Word2Vec(gutenberg.sents(), min_count = 5, size = 100)
            pickle.dump(self.model, open('model.bin', 'wb'))
            print('done reading.')

    def write(self, lines):
        start_word = numpy.random.rand(100)
        target_word = numpy.random.rand(100)
        difference = start_word - target_word

        title = self.model.most_similar(positive=[start_word], topn=1)[0][0]

        neighbor_words = self.model.most_similar(positive=[title], topn=lines)
        target_neighbor_words = []
        for word, similarity in neighbor_words:
            target_word = self.model.most_similar(positive=[word], topn=1)
            target_neighbor_words.append(target_word[0])

        print(utils.capitalize(title))
        first = True
        for x, y in zip(neighbor_words, target_neighbor_words):
            print(self.templatize(x[0], y[0], first))
            first = False

    def templatize(self, x, y, first = False):
        """
        Takes in two words and produces a templatized str

        Args:
            x: first word
            y: second word
            first: if True, template is 'if _x_ were _y_';
                   if False, template is '_x_ would be _y_'
        """
        if first:
            return 'if {} were {}'.format(x, y)
        else:
            return '{} would be {}'.format(x, y)
        
        
        
    
