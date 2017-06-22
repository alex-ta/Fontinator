import random as rand

class WordDict:
    """
    Allows creating a random sentence.
    """

    def __init__(self, word_dict: list):
        """
        Creates an WordDict object from a list of words.
        which will be used as random word source
        :param word_dict: A list of words
        """
        self._ger_word_dict = word_dict

    def load_from_textfile(input_file_path: str, enc="UTF8"):
        """
        Creates an WordDict object from an text file.
        :param input_file_path: The path to the text file
        :param enc: The encoding of the text file.
        :return: A WordDict object
        """
        ger_word_dict = []
        with open(input_file_path, encoding=enc) as f:
            for line in f:
                words = line.split(sep=' ')
                words[len(words) - 1] = words[len(words) - 1].replace('\n', '')
                ger_word_dict.extend(words)
        return WordDict(ger_word_dict)

    def get_sentence(self, word_count: int):
        """
        Creates an sentence with <word_count> words
        :param word_count: The number of words in the returned sentence
        :return: A string containing random words
        """
        sentence = ""
        for i in range(word_count):
            r_int = rand.randint(0, len(self._ger_word_dict) - 1)
            rand_word = self._ger_word_dict[r_int]
            sentence += rand_word + " "
        return sentence

    def get_word_count(self):
        """
        Returns the size of word dict
        :return: The size of words in the WordDict
        """
        return len(self._ger_word_dict)
