import csv
import string
import re
import math
import sys
import collections
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
from scipy.stats import pearsonr


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
    X_validate = []
    cutoff = math.floor(len(strvec_list)*0.8)
    validation_cutoff = math.floor(len(strvec_list)*0.9)
    for i in range(1, len(strvec_list)):
        if i < cutoff:
            X_train.append(strvec_list[i])
        elif i < validation_cutoff:
            X_validate.append(strvec_list[i])
        else:
            X_test.append(strvec_list[i])

    # print('Maximum review length: {}'.format(
    #     len(max((X_train + X_test), key=len)))) Max review length = 116 words

    max_words = 120
    vocabulary_size = len(words)
  
    X_train_valence = sequence.pad_sequences(X_train, maxlen = max_words)
    X_test_valence = sequence.pad_sequences(X_test, maxlen = max_words)
    X_validate = sequence.pad_sequences(X_validate, maxlen = max_words)


    y_train_valence = [float(emobank[i][1])/5 for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_valence = [float(emobank[i][1])/5 for i in range(math.floor(0.9*len(strvec_list)), len(strvec_list))]
    y_validate_valence = [float(emobank[i][1])/5 for i in range(math.floor(0.8*len(strvec_list)), math.floor(len(strvec_list)*0.9))]


    y_train_arousal = [float(emobank[i][2])/5 for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_arousal = [float(emobank[i][2])/5 for i in range(math.floor(0.9 * len(strvec_list)), len(strvec_list))]
    y_validate_arousal = [float(emobank[i][2])/5 for i in range(math.floor(0.8*len(strvec_list)), math.floor(len(strvec_list)*0.9))]

    y_train_dominance = [float(emobank[i][3])/5 for i in range(1, math.floor(0.8 * len(strvec_list)))]
    y_test_dominance = [float(emobank[i][3])/5 for i in range(math.floor(0.9 * len(strvec_list)), len(strvec_list))]
    y_validate_dominance = [float(emobank[i][3])/5 for i in range(math.floor(0.8*len(strvec_list)), math.floor(len(strvec_list)*0.9))]



#    min = sys.float_info.max
#    optimalLSTM = 0
#    for i in range(1, 10): #LSTM should have 90 neurons for dominance
#        lstm = 50 + i*10
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(lstm))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalLSTM = lstm
#    print(optimalLSTM)
#
#
#    min = sys.float_info.max
#    optimalDropout = 0
#    for i in range(1, 5): #Dropout layer should be 0.4 for dominance
#        dropout = i*0.1
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(dropout))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalDropout = dropout
#    print(optimalDropout)

#    min = sys.float_info.max
#    optimalEmbedding = 0
#    for i in range(1, 5): #46 is best embedding size for dominance
#        embedding_size = 38 + i*2
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalEmbedding = embedding_size
#            print(optimalEmbedding)
#    print(embedding_size)

#    min = sys.float_info.max
#    optimalLSTM = 0
#    for i in range(1, 10): #LSTM should have 90 neurons for arousal
#        lstm = 50 + i*10
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(lstm))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_arousal[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_arousal[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_arousal, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalLSTM = lstm
#    print(optimalLSTM)
#
#
#    min = sys.float_info.max
#    optimalDropout = 0
#    for i in range(1, 5): #Dropout layer should be 0.3 for arousal
#        dropout = i*0.1
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(dropout))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalDropout = dropout
#    print(optimalDropout)
#
#    min = sys.float_info.max
#    optimalEmbedding = 0
#    for i in range(1, 5): #52 is best embedding size for arousal
#        embedding_size = 44 + i*2
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalEmbedding = embedding_size
#            print(optimalEmbedding)
#    print(embedding_size)


#    min = sys.float_info.max
#    optimalLSTM = 0
#    for i in range(1, 10): #LSTM should have 90 neurons for arousal
#        lstm = 50 + i*10
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(lstm))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_arousal[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_arousal[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_arousal, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalLSTM = lstm
#    print(optimalLSTM)
#
#
#    min = sys.float_info.max
#    optimalDropout = 0
#    for i in range(1, 5): #Dropout layer should be 0.3 for arousal
#        dropout = i*0.1
#        embedding_size = 35
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(dropout))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalDropout = dropout
#    print(optimalDropout)
#
#    min = sys.float_info.max
#    optimalEmbedding = 0
#    for i in range(1, 5): #52 is best embedding size for arousal
#        embedding_size = 44 + i*2
#        model_dominance = Sequential()
#        model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
#        model_dominance.add(LSTM(100))
#        model_dominance.add(Dropout(0.2))
#        model_dominance.add(Dense(1, activation='sigmoid'))
#        print(model_dominance.summary())
#        model_dominance.compile(loss='mean_squared_error',
#                        optimizer='adam')
#        batch_size = 2000
#        num_epochs = 3
#        X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
#        X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
#        model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
#        scores = model_dominance.evaluate(X_validate, y_validate_dominance, verbose=0)
#        if 25*scores < min:
#            min = 25*scores
#            optimalEmbedding = embedding_size
#            print(optimalEmbedding)
#    print(embedding_size)

    HP_EMBEDDING = hp.HParam('embedding', hp.Discrete([16, 100]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
    HP_LSTM = hp.HParam('LSTM', hp.Discrete([80, 150]))
    
    RSS = 'mean_squared_error'
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
        hparams=[HP_EMBEDDING, HP_DROPOUT, HP_LSTM],
        metrics=[hp.Metric(RSS, display_name='Mean Squared Error')]
                          )

    session_num = 0

    for num_units in HP_EMBEDDING.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for num_lstm in HP_LSTM.domain.values:
                hparams = {
                    HP_EMBEDDING: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_LSTM: num_lstm
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1

    embedding_size = 46
    model_dominance = Sequential()
    model_dominance.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
    model_dominance.add(LSTM(90))
    model_dominance.add(Dropout(0.4))
    model_dominance.add(Dense(1, activation='sigmoid'))
    print(model_dominance.summary())

    model_dominance.compile(loss='mean_squared_error',
              optimizer='adam')

    batch_size = 2000
    num_epochs = 13
    X_valid, y_valid = X_train_valence[:batch_size], y_train_dominance[:batch_size]
    X_train2, y_train2 = X_train_valence[batch_size:], y_train_dominance[batch_size:]
    model_dominance.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    count = 0
    scores = model_dominance.evaluate(X_test_valence, y_test_dominance, verbose=0)
    print(25*scores)
    average_loss = 0
    for i in range(len(X_test_valence)):
        prediction = 5*model_dominance.predict(np.array([X_test_valence[i]]))[0][0]
        difference = abs(prediction - 5*y_test_dominance[i])
        average_loss += difference
        if difference <= 0.3:
            count += 1
    print("Classification accuracy (dominance): " + str(count / len(X_test_valence)))
    print("Average absolute error (dominance): " + str(average_loss / len(X_test_valence)))


    embedding_size = 52
    model_arousal = Sequential()
    model_arousal.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
    model_arousal.add(LSTM(90))
    model_arousal.add(Dropout(0.3))
    model_arousal.add(Dense(1, activation='sigmoid'))
    print(model_arousal.summary())

    model_arousal.compile(loss='mean_squared_error',
                  optimizer='adam')

    X_valid, y_valid = X_train_valence[:batch_size], y_train_arousal[:batch_size]
    X_train2, y_train2 = X_train_valence[batch_size:], y_train_arousal[batch_size:]
    model_arousal.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    count = 0
    scores = model_arousal.evaluate(X_test_valence, y_test_arousal, verbose=0)
    print(25*scores)
    average_loss = 0
    for i in range(len(X_test_valence)):
        prediction = 5*model_arousal.predict(np.array([X_test_valence[i]]))[0][0]
        difference = abs(prediction - 5*y_test_arousal[i])
        average_loss += difference
        if difference <= 0.3:
            count += 1
    print("Classification accuracy (arousal): " + str(count / len(X_test_valence)))
    print("Average absolute error (arousal): " + str(average_loss / len(X_test_valence)))

    embedding_size = 35
    model_valence = Sequential()
    model_valence.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
    model_valence.add(LSTM(115))
    model_valence.add(Dropout(0.1))
    model_valence.add(Dense(1, activation='sigmoid'))
    print(model_valence.summary())
    
    model_valence.compile(loss='mean_squared_error',
                          optimizer='adam')

    X_valid, y_valid = X_train_valence[:batch_size], y_train_valence[:batch_size]
    X_train2, y_train2 = X_train_valence[batch_size:], y_train_valence[batch_size:]
    model_valence.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    count = 0
    scores = model_valence.evaluate(X_test_valence, y_test_valence, verbose=0)
    print(25*scores)
    average_loss = 0
    for i in range(len(X_test_valence)):
        prediction = 5*model_valence.predict(np.array([X_test_valence[i]]))[0][0]
        difference = abs(prediction - 5*y_test_valence[i])
        average_loss += difference
        if difference <= 0.3:
            count += 1
    print("Classification accuracy (valence): " + str(count / len(X_test_valence)))
    print("Average absolute error (valence): " + str(average_loss / len(X_test_valence)))
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
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        print(hparams[HP_EMBEDDING])
        error = train_test_model(hparams)
        tf.summary.scalar(mean_squared_error, error, step=1)
def train_test_model(hparams):
    embedding_size = hparams[HP_EMBEDDING]
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, embedding_size, input_length=max_words))
    model.add(LSTM([HP_LSTM]))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
                  optimizer='adam',
                  loss='mean_squared_error',
                  )
                                            
    model.fit(x_train, y_train, epochs=3)
    return 25*model.evaluate(x_test, y_test, verbose = 0)
def process_string(sentence):
    """
    Processes string into a vector of integers corresponding to words in order
    so that the RNN will function properly.
    """
    result = sentence.split(" ")
    for i in range(len(result)):
        word = result[i]
        if word not in words:
            numWords = len(words)
            words[word] = numWords + 1
        result[i] = words[word]
    return result

if __name__ == '__main__':
    main()
