import csv
import string
import re
import math
import collections
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout


strip_table = str.maketrans('', '', string.punctuation)
non_ascii = re.compile(r'[^\x00-\x7F]+')
BIAS_FEATURE = '##'

words = collections.defaultdict(int)
vocabulary_size = 0

def main():
    emobank = read_csv('EmoBank-master/corpus/emobank.csv')
    strvec_list = [process_string(row[4]) for row in emobank]
    strvec_list.pop(0)

    X_train = []
    X_test = []
    cutoff = math.floor(len(strvec_list)*0.8)
    for i in range(1, len(strvec_list)):
        if i < cutoff:
            X_train.append(strvec_list[i])
        else:
            X_test.append(strvec_list[i])

    # print('Maximum review length: {}'.format(
    #     len(max((X_train + X_test), key=len)))) Max review length = 116 words

    max_words = 120
    vocabulary_size = len(words)
    X_train_valence = sequence.pad_sequences(X_train, maxlen = max_words)
    X_test_valence = sequence.pad_sequences(X_test, maxlen = max_words)


    embedding_size = 32
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    y_train_valence = [float(emobank[i][1]) for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_valence = [float(emobank[i][1]) for i in range(math.floor(0.8*len(strvec_list)), len(strvec_list))]

    y_train_arousal = [float(emobank[i][2])/5 for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_arousal = [float(emobank[i][2])/5 for i in range(math.floor(0.8 * len(strvec_list)), len(strvec_list))]

    y_train_dominance = [float(emobank[i][3]) for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_dominance = [float(emobank[i][3]) for i in range(math.floor(0.8 * len(strvec_list)), len(strvec_list))]


    batch_size = 64
    num_epochs = 3
    X_valid, y_valid = X_train_valence[:batch_size], y_train_arousal[:batch_size]
    X_train2, y_train2 = X_train_valence[batch_size:], y_train_arousal[batch_size:]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    scores = model.evaluate(X_test_valence, y_test_arousal, verbose=0)
    print(scores)




def read_csv(path):
    with open(path) as datafile:
        reader = csv.reader(datafile)
        return [row for row in reader]

def RnMean(vectors):
    w = len(vectors)
    mean = collections.defaultdict(float)
    for vec in vectors:
        for f, v in vec.items():
            mean[f] += v
    return collections.defaultdict(float, ((f, v / w) for f, v in mean.items()))

def process_string(sentence):
    """
    Takes the string in the last column of the example data, and returns the string
    """
    result = sentence.split(" ")
    for i in range(len(result)):
        word = result[i]
        if word not in words:
            numWords = len(words)
            words[word] = numWords + 1
        result[i] = words[word]
    return result
    # counts = collections.defaultdict(int)
    # s = non_ascii.sub(' ', sentence)
    # for token in s.translate(strip_table).lower().split():
    #     counts[token] += 1
    # counts[BIAS_FEATURE] = 1
    #return counts

if __name__ == '__main__':
    main()