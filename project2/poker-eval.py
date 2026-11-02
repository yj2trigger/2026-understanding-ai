## -------------------------------------------------
import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd

# load the model from disk
filename = 'poker-model3.sav'
model3 = pickle.load(open(filename, 'rb'))

filename = 'poker-model4.sav'
model4 = pickle.load(open(filename, 'rb'))
 
# given parameters:
#   hand: list of tuples (suit, rank); length is 3 or 4
#   opposite_score: the score of opposite player
#   
# return:
#   True if you want to call,
#   False if you want to fold
def predict_call(hand, opposite_score):
    inp = list()
    for suit, rank in hand:
        inp.extend([suit, rank])

    if len(hand) == 3:
        X_test = pd.DataFrame(
            [inp], 
            columns=['Suit of Card 1', 'Rank of Card 1', 
                     'Suit of Card 2', 'Rank of Card 2', 
                     'Suit of Card 3', 'Rank of Card 3'])
        y_pred = model3.predict_proba(X_test)

    else:
        X_test = pd.DataFrame(
            [inp], 
            columns=['Suit of Card 1', 'Rank of Card 1', 
                     'Suit of Card 2', 'Rank of Card 2', 
                     'Suit of Card 3', 'Rank of Card 3', 
                     'Suit of Card 4', 'Rank of Card 4'])
        y_pred = model4.predict_proba(X_test)

    # compute expectation
    exp = sum([i * p for i, p in enumerate(y_pred[0])])
    # print("expected score = ", exp)
    if exp >= opposite_score:
        return True
    return False
## -------------------------------------------------

import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import random

test = pd.read_csv('poker-hand-testing.csv')

money = [1000] * 10
for i in range(10):
    samp = np.random.randint(0, len(test), size=1000)
    X_test = test.iloc[samp, :-1].to_numpy()
    y_test = test.iloc[samp, -1].to_numpy()

    for test_hand, test_score in zip(X_test, y_test):
        test_input = [ (suit, rank) for suit, rank in zip(test_hand[::2], test_hand[1::2]) ]
        oppo_score = y_test[random.randrange(len(y_test))]

        if not predict_call(test_input[:3], oppo_score):
            money[i] -= 1
            continue

        if not predict_call(test_input[:4], oppo_score):
            money[i] -= 2
            continue

        if test_score > oppo_score:
            money[i] += 3

        elif test_score < oppo_score:
            money[i] -= 3
        # if tie, your money stays the same

print('average of money= {}'.format(round(np.mean(money), 3)))