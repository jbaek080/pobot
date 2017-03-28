from gensim.models import Word2Vec as Word2Vec
from nltk.corpus import gutenberg as gutenberg
from nltk.corpus import brown as brown
from nltk import pos_tag
from nltk import word_tokenize
import pickle
import numpy
import logging
import utils
import math
import random
from typing import List, Tuple

class Bot:
    def __init__(self, lines = 10, overread = False):
        numpy.random.seed()
        random.seed()

        self.read(overread = overread)
        self.write(lines)

    def read(self, overread = False):
        """ Trains on some sentences to yield Word2Vec model, or loads
        previously pickled model

        Args:
            overread: if True, ignore model.bin, and read all over again
        """
        try:
            print('trying to remember...')
            self.model = pickle.load(open('model.bin', 'rb'))
            print('remembered previous reading session.')
        except Exception as e:
            print('couldn\'t remember because', e)
            print('reading...')
            self.model = Word2Vec(gutenberg.sents(), min_count = 1, size = 100)
            pickle.dump(self.model, open('model.bin', 'wb'))
            print('done reading.')

    def write(self, lines):
        """ Prints out poem
        """
        title, title_vector = self.get_random_word_vector()

        print(utils.capitalize(title))
        start_word = title
        for _ in range(lines):
            start_word = self.write_couplet_and_return_next_start(start_word)

    def write_couplet_and_return_next_start(self, start_word):
        """ Writes couplet of form::
            if __ were __,
            then __ would be __ (because __)

        Args:
            start_vector: vector of word to fill first slot
        """
        start_vector = self.model[start_word]
        target_word = ''
        diff_confidence = 0
        while (not target_word 
                or self.pos(start_word) != self.pos(target_word)
                or diff_confidence < 0.5):
            target_word, target_vector = self.get_random_word_vector(
                        closeto=start_word)
            diff_vector = start_vector - target_vector
            diff_word, diff_confidence = self.vec_to_word(diff_vector)
         
        print(self.if_x_were_y(start_word, target_word, diff_word))
        print(self.then_a_would_be_b(diff_vector))
        return target_word

    def get_random_word_vector(self, closeto=None) -> Tuple[str, List[float]]:
        """ Gets a random word vector from model.

        If closeto is specified, returns a word vector that is close to that.
        
        Returns:
            a random (word: str, vec: List[float]) tuple
        """
        if closeto is not None:
            word = random.choice(
                self.model.most_similar(positive=[closeto], topn=5))[0]
        else:
            word = random.choice(list(self.model.vocab.items()))[0]
        return (word, self.model[word])

    def pos(self, word):
        return pos_tag(word_tokenize(word), tagset='universal')[0][1]

    def then_a_would_be_b(self, difference, how_many_more = 3):
        if how_many_more == 0:
            return ''
        first_word, first_vector = self.get_random_word_vector(
                                                    closeto=difference)
        second_vector = first_vector - difference
        second_word = self.vec_to_word(second_vector)

        return ('then {} would be {}\n'.format(
                first_word, 
                second_word) 
                + self.then_a_would_be_b(
                        second_vector, 
                        how_many_more - 1))
        

    def vec_to_word(self, vec: List[float]) -> str:
        """ Takes in vector and returns word
        
        Returns:
            word, a str
        """
        return self.model.most_similar(positive=[vec], topn=1)[0]

    def if_x_were_y(self, x, y, z):
        """
        Takes in two words and produces a templatized str

        Args:
            x: first word
            y: second word
            z: third word
        """
        return 'if {} were {}, because of {},'.format(x, y, z)
        
        
        
    
