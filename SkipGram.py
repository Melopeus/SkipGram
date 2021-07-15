import tensorflow as tf
import numpy as np
import re
import math
import tensorflow.keras.backend as kb
from tensorflow.python.platform.tf_logging import flush
from SkipGramTokenizer import SkipGramTokenizer


class SkipGramModel(tf.keras.Model):

    def __init__(self, word_size, hidden_size):
        super(SkipGramModel, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            hidden_size, use_bias=False, activation=tf.keras.activations.linear, input_shape=[word_size])
        self.output_layer = tf.keras.layers.Dense(
            word_size, use_bias=False, activation=tf.keras.activations.linear)

    def call(self, inputs):
        print(inputs)
        x = self.hidden_layer(inputs)
        return self.output_layer(x)


class SkipGram:
    def __init__(self, context_size) -> None:
        self.model = None
        self.context_size = context_size
        self.training_data = None
        self.words_index_map = None
        self.index_words_map = None

    def init_from_file(self, path: str):
        tokenizer = SkipGramTokenizer()
        with open(path, 'r') as training_text_file:
            raw_text = training_text_file.read()
            self.words_index_map, self.training_data = tokenizer.get_training_data(raw_text, self.context_size)
            self.index_words_map = { value : key for (key, value) in self.words_index_map.items()}

        word_count = len(self.words_index_map)
        self.model = SkipGramModel(word_count, word_count)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=self.__lossFunction)
        

    def init_from_string(self, raw_text: str):
        tokenizer = SkipGramTokenizer()
        self.words_index_map, self.training_data = tokenizer.get_training_data(raw_text, self.context_size)
        word_count = len(self.words_index_map)
        self.model = SkipGramModel(word_count, word_count//2)
        self.model.compile(optimizer='adam', loss=self.__lossFunction)
        

    def train(self):
        x = []
        y = []
        for pair in self.training_data:
            x.append(pair[0])
            y.append(pair[1])
        
        x = np.array(x)
        y = np.array(y)
        self.model.fit(x, y, epochs= 50)

    def predict(self):
        pass

    def __lossFunction(self, actual_output, u):
        return -kb.sum(actual_output*u) + len(self.words_index_map) * kb.log(kb.sum(kb.exp(u)))



if __name__ == "__main__":
    SG = SkipGram(2)
    SG.init_from_file("text.txt")
    SG.train()