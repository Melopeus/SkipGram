import numpy as np
import re
import math

class SkipGramTokenizer:
    def __init__(self, language="english") -> None:
        with open('stopwords/'+language, 'r', encoding='utf-8') as stop_words_file:
            self.stop_words = stop_words_file.read()
            self.stop_words = set(re.split('\n', self.stop_words))

    def __tokenize(self, text: str, context_size: int):
        text = text.lower()
        text = text.replace(",", "")
        text = text.replace(":", "")
        text = text.replace("'", "")
        sentances = re.split(r"[.:!?\n]", text)
        sentances = map(lambda sentence: 
                        list(filter(
                                    lambda word: False if word == "" or word in self.stop_words else True,
                                    sentence.split(" ")
                                    )
                            ), sentances
                        )
        sentances = list(filter(lambda sentance: len(sentance) >= context_size + 1, sentances))
        return list(sentances)

    def __get_all_words(self, sentences: list):
        all_words_set = set()
        for sentence in sentences:
            for word in sentence:
                all_words_set.add(word)

        all_words_list = list(all_words_set)
        all_words_list = sorted(all_words_list)

        all_words_index_map = dict()
        for index, word in enumerate(all_words_list):
            all_words_index_map.update({word : index})

        return all_words_index_map

    @staticmethod
    def __find_context(center_word_index, sentence, context_size):
        context_words_found = 0
        current_context_word_index = 0

        if center_word_index + math.ceil(context_size/2) >= len(sentence):
            current_context_word_index = -context_size + len(sentence) - 1
        else:
            current_context_word_index = center_word_index - context_size//2

        context = []
        while context_words_found < context_size:
            if current_context_word_index < 0:
                current_context_word_index += 1
                continue
            if current_context_word_index == center_word_index:
                current_context_word_index += 1
                continue

            context.append(sentence[current_context_word_index])
            current_context_word_index += 1
            context_words_found += 1
        
        return context

    def get_training_data(self, text: str, context_size: int):
        sentences = self.__tokenize(text, context_size)
        words_index_map = self.__get_all_words(sentences)
        words_count = len(words_index_map)

        word_context_pairs = []
        CENTER_WORD_VECTOR = 0
        CONTEXT_VECTOR = 1
        for sentence in sentences:
            for center_word_index, center_word in enumerate(sentence):
                pair = [np.zeros(words_count), np.zeros(words_count)]
                context = self.__find_context(center_word_index, sentence, context_size)
                context = list(map(lambda word : words_index_map[word], context))

                pair[CENTER_WORD_VECTOR][words_index_map[center_word]] = 1
                #for context_index in context:
                #    pair[CONTEXT_VECTOR][context_index] += 1

                context_onehot_array = []
                for context_index in context:
                    current_context_onehot_word = [ 1 if k == context_index else 0 for k in range(words_count)]
                    context_onehot_array.append(np.array(current_context_onehot_word))
                
                pair[CONTEXT_VECTOR] = np.sum(context_onehot_array, axis=0)  

                word_context_pairs.append(pair)

        return words_index_map, word_context_pairs
    
    def get_training_data_explicit(self, text: str, context_size: int):
        """
        This Function return pairs of actual vectors in the context for a word.
        """
        sentences = self.__tokenize(text, context_size)
        words_index_map = self.__get_all_words(sentences)
        words_count = len(words_index_map)

        word_context_pairs = []
        CENTER_WORD_VECTOR = 0
        CONTEXT_VECTOR = 1
        for sentence in sentences:
            for center_word_index, center_word in enumerate(sentence):
                pair = [np.zeros(words_count), np.zeros(words_count)]
                context = self.__find_context(center_word_index, sentence, context_size)
                context = list(map(lambda word : words_index_map[word], context))

                pair[CENTER_WORD_VECTOR][words_index_map[center_word]] = 1
                context_onehot_array = []
                for context_index in context:
                    current_context_onehot_word = [ 1 if k == context_index else 0 for k in range(words_count)]
                    context_onehot_array.append(np.array(current_context_onehot_word))

                pair[CONTEXT_VECTOR] = context_onehot_array    

                #word_context_pairs.append((words_index_map[center_word], context))
                word_context_pairs.append(pair)

        return words_index_map, word_context_pairs

if __name__ == "__main__":
    tokenizer = SkipGramTokenizer()
    tokenizer.get_training_data_explicit("natural language processing and machine learning is fun and exciting", 2)
    #print(tokenizer.get_training_data("Private methods are those that should neither be accessed the class nor by any base class. In Python, there is no of Private methods that cannot be accessed except inside a class. However, to define a private method prefix the member name with.", 4))
    x=[(1,2),(1,2),(1,2)]
    print(*list(zip(*x)))