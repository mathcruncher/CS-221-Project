import collections
import csv
import string
import re
import math

strip_table = str.maketrans('', '', string.punctuation)
#strip_table = str.maketrans('', '', '0123456789[]<>*#:@.,;+$%()-"\\/')
non_ascii = re.compile(r'[^\x00-\x7F]+')
BIAS_FEATURE = '##'

def main():
    # V, A, D
    emobank = read_csv('emobank.txt')
    strvec_list = [process_string(row[4]) for row in emobank]
    
    word_mean = RnMean(strvec_list)
    with open('multireg_ctr.txt', 'wt', 1) as mean_out:
        for k, v in sorted(word_mean.items()):
            mean_out.write('{} {}\n'.format(k, v))
    
    valence_set = [(strvec_list[i], float(emobank[i][1])) for i in range(len(emobank))]
    arousal_set = [(strvec_list[i], float(emobank[i][2])) for i in range(len(emobank))]
    dominance_set = [(strvec_list[i], float(emobank[i][3])) for i in range(len(emobank))]
    
    print('Training Valence')
    valence_weights = multipleRegression(100, .01, valence_set)
    print('Training Arousal')
    arousal_weights = multipleRegression(100, .01, arousal_set)
    print('Training Dominance')
    dominance_weights = multipleRegression(100, .01, dominance_set)

    with open('multireg_vad.txt', 'wt', 1) as weightfile:
        for token in sorted(valence_weights.keys()):
            weightfile.write('{}, {}, {}, {}\n'.format(
                token,
                valence_weights[token],
                arousal_weights[token],
                dominance_weights[token]
            ))
    
    try:
        while True:
            query_vec = process_string(input())
            v = dotProduct(valence_weights, query_vec)
            a = dotProduct(arousal_weights, query_vec)
            d = dotProduct(dominance_weights, query_vec)
            print('Valence: {}\nArousal: {}\nDominance: {}\n'.format(v, a, d))
    except EOFError:
        pass
        
def recenter(vectors, centroid):
    adjusted = [collections.defaultdict(float, ex) for ex in vectors]
    for adj in adjusted:
        increment(adj, -1, centroid)
    return adjusted

def RnMean(vectors):
    w = len(vectors)
    mean = collections.defaultdict(float)
    for vec in vectors:
        for f, v in vec.items():
            mean[f] += v
    return collections.defaultdict(float, ((f, v / w) for f, v in mean.items()))

def multipleRegression(steps, step_size, training_set):
    weights = {}
    for i in range(1, steps + 1):
        #print('Step size: {}'.format(1.0 / i))
        r = gradTrainLoss(weights, training_set)
        #increment(weights, -step_size, r)
        increment(weights, -(1.0 / i), r)
    return weights

def gradTrainLoss(weights, training_set):
    """
    Implements the gradient of squared loss on multivariable linear regression
    for batch gradient descent
    @param dict weights: a weight vector
    @param list training_set: a list of training examples, tuple of (dict(x), y)
    @return float: the gradient of the training loss function
    """
    grad = collections.defaultdict(float)
    trainLoss = 0
    for example in training_set:
        residual = dotProduct(weights, example[0]) - example[1] # w . x - y
        trainLoss += (residual ** 2)
        #for feature, value in example[0].items():
        #    grad[feature] += residual * value
        increment(grad, residual, example[0])
    fac = 2.0 / len(training_set)
    print('Train Loss: {}'.format(trainLoss / len(training_set)))
    return collections.defaultdict(float, ((f, fac * v) for f, v in grad.items()))

def read_csv(path):
    with open(path) as datafile:
        reader = csv.reader(datafile)
        return [row for row in reader]
        
def process_string(sentence):
    """
    Takes the string in the last column of the example data, and puts into a dictionary with counts.
    @param str sentence: the example string from the data
    @return dict: sparse vector of word counts
    """
    counts = collections.defaultdict(int)
    s = non_ascii.sub(' ', sentence)
    for token in s.translate(strip_table).lower().split():
        counts[token] += 1
    counts[BIAS_FEATURE] = 1
    return counts

'''
dotProduct() and increment() taken from util.py of assignment 2
'''        
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale
        
if __name__ == '__main__':
    main()