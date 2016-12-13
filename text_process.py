import nltk
from sklearn.preprocessing import OneHotEncoder
from nltk.data import load
import numpy as np

arpabet = nltk.corpus.cmudict.dict()
word_type_dict = {}
i = 1
for tag in load('help/tagsets/upenn_tagset.pickle').keys():
    word_type_dict[tag] = i
    i = i + 1

paus_type = i;

phoneme_dict = {'PAU': 0, 'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39}
phoneme_enc = OneHotEncoder()
phoneme_values = np.asarray(list(phoneme_dict.values())).reshape(-1,1)
phoneme_enc.fit(phoneme_values)

class Phoneme:
    name = 'AA'
    phoneme_type = 0
    stress = 3
    position_in_word = 0
    word_size = 0
    word_number = 0

class Word:
    number_of_phonemes = 0
    word_number = 0

def create_pau(word_number):
    phoneme = Phoneme()
    phoneme_type = phoneme_dict['PAU']
    phoneme.stress = 0
    phoneme.name = 'PAU'
    phoneme.position_in_word = 0
    phoneme.word_size = 0
    phoneme.word_number = word_number
    return phoneme

def generate_tags(sentence):
    tags = []
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    phonemes = []
    primary_phonemes = []
    words_with_data = []

    pos_sentence = 0
    pos_phoneme_in_sentence = 0
    for i in range(len(words)-1):
        if words[i].lower() in arpabet.keys():
            pos_word = 0
            word_phonemes = arpabet[words[i].lower()]
            for j in range(len(word_phonemes[0])):
                if any((c in phoneme_dict.keys()) for c in word_phonemes[0][j][0:-1]) or any((c in phoneme_dict.keys()) for c in word_phonemes[0][j]):
                    phoneme = Phoneme()
                    phoneme_type = ''
                    if word_phonemes[0][j][-1] in '012':
                        phoneme.stress = int(word_phonemes[0][j][-1])
                        phoneme_type = word_phonemes[0][j][0:-1]
                    else:
                        phoneme.stress = 3
                        phoneme_type = word_phonemes[0][j]

                    phoneme.name = phoneme_type
                    phoneme.phoneme_type = phoneme_dict[phoneme_type]

                    phoneme.position_in_word = pos_word
                    pos_word = pos_word + 1

                    phoneme.word_size = len(word_phonemes[0])

                    phoneme.word_number = i
                    phonemes.append(phoneme)

                    if phoneme.stress == 1:
                        primary_phonemes.append(len(phonemes) - 1)


        word = Word()
        word.word_number = i
        word.number_of_phonemes = pos_word + 1
        words_with_data.append(word)

    for i in range(len(phonemes)):
        tag = []
        phoneme = phonemes[i]

        kvint_types = []
        if i < 1: kvint_types.append(0)
        else: kvint_types.append(phonemes[i-1].phoneme_type)
        if i < 2: kvint_types.append(0)
        else: kvint_types.append(phonemes[i-2].phoneme_type)
        kvint_types.append(phoneme.phoneme_type)
        if i > len(phonemes)-4: kvint_types.append(0)
        else: kvint_types.append(phonemes[i+1].phoneme_type)
        if i > len(phonemes)-3: kvint_types.append(0)
        else: kvint_types.append(phonemes[i+2].phoneme_type)

        for phoneme_type in kvint_types:
            for onehot_value in phoneme_enc.transform([[phoneme_type]]).toarray()[0]:
                if (onehot_value == 0):
                    tag.append(0.01)
                else:
                    tag.append(0.99)

        tag.append(phoneme.stress)

        tag.append(phoneme.position_in_word)
        tag.append(phoneme.word_size - phoneme.position_in_word -1)

        j = 0
        while j < len(primary_phonemes) and primary_phonemes[j] < i:
            j = j+1

        j = j-1

        if len(primary_phonemes) <= 0 or i <= primary_phonemes[j] or j == -1: tag.append(0)
        else: tag.append(i - primary_phonemes[j])

        if (j > len(primary_phonemes) - 2): tag.append(0)
        else: tag.append(primary_phonemes[j+1] - i)


        word_number = phoneme.word_number
        tag.append(word_type_dict[tagged[word_number][1]])

        tag.append(word_number)
        tag.append(len(words) - word_number - 1)

        tag.append(words_with_data[word_number].number_of_phonemes)
        if word_number == 0: tag.append(0)
        else: tag.append(words_with_data[word_number - 1].number_of_phonemes)
        if word_number == len(words) - 2: tag.append(0)
        else: tag.append(words_with_data[word_number+1].number_of_phonemes)


        tag.append(len(words))

        tag.append(len(phonemes))


        tags.append([phoneme.name,tag])

    return tags
