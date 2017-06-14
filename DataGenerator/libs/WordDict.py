import random as rand

class WordDict:
    def __init__(self, word_dict: list):
        self._ger_word_dict = word_dict

    # Creates an WordDict object
    def load_from_textfile(input_file_path: str, enc="UTF8"):
        ger_word_dict = []
        with open(input_file_path, encoding=enc) as f:
            for line in f:
                words = line.split(sep=' ')
                words[len(words) - 1] = words[len(words) - 1].replace('\n', '')
                ger_word_dict.extend(words)
        return WordDict(ger_word_dict)

    # Creates an sentence with <word_count> words
    def get_sentence(self, word_count: int):
        sentence = ""
        for i in range(word_count):
            r_int = rand.randint(0, len(self._ger_word_dict) - 1)
            rand_word = self._ger_word_dict[r_int]
            sentence += rand_word + " "
        return sentence

    # Returns the size of word dict
    def get_word_count(self):
        return len(self._ger_word_dict)
