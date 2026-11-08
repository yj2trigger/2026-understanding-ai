## -------------------------------------------------
import pickle
from collections import Counter
import pandas as pd

# 1. 단일 랜덤 포레스트 모델 로드 (poker-hand-model.sav)
filename = 'poker-hand-model.sav'
with open(filename, 'rb') as f:
    model = pickle.load(f)
classes = model.classes_

# 2. 족보 확률 예측 모델용 특성 함수 (features.py 내장)
STRAIGHT_SETS = [frozenset(range(low, low + 5)) for low in range(1, 10)]
STRAIGHT_SETS.append(frozenset((1, 10, 11, 12, 13)))

ROYAL_RANKS = frozenset((1, 10, 11, 12, 13))

FEATURE_COLUMNS = [
    "n_cards",           # 카드 장수 (3 또는 4)
    "rank_max_count",    # 같은 숫자의 최대 장수
    "pair_count",        # 페어 개수
    "triple_count",      # 트리플 개수
    "rank_unique",       # 서로 다른 숫자 개수
    "suit_max_count",    # 같은 문양의 최대 장수
    "suit_unique",       # 서로 다른 문양 개수
    "straight_windows",  # 현재 숫자를 모두 포함하는 STRAIGHT_SETS 개수
    "sf_windows",        # 모두 같은 문양이면 straight_windows, 아니면 0
    "royal_possible",    # 같은 문양 + 숫자가 모두 {A,10,J,Q,K}
    "current_score",     # hand_category(hand)
]

def hand_category(hand):
    """3~5장의 현재 족보 점수 (0 노페어 ~ 9 로열 플러시)."""
    if len(hand) not in (3, 4, 5):
        raise ValueError("카드는 3장, 4장 또는 5장이어야 합니다.")

    ranks = [rank for suit, rank in hand]
    suits = [suit for suit, rank in hand]
    frequencies = sorted(Counter(ranks).values(), reverse=True)

    if len(hand) == 5:
        rank_set = frozenset(ranks)
        straight = len(rank_set) == 5 and rank_set in STRAIGHT_SETS
        flush = len(set(suits)) == 1

        if straight and flush:
            return 9 if rank_set == ROYAL_RANKS else 8
        if 4 in frequencies:
            return 7
        if frequencies == [3, 2]:
            return 6
        if flush:
            return 5
        if straight:
            return 4

    if 4 in frequencies:
        return 7
    if 3 in frequencies:
        return 3
    if frequencies.count(2) == 2:
        return 2
    if 2 in frequencies:
        return 1
    return 0

def extract_features(hand):
    """(suit, rank) 튜플 3~4개 → FEATURE_COLUMNS 순서의 정수 리스트."""
    if len(hand) not in (3, 4):
        raise ValueError("현재 카드가 3장 또는 4장이어야 합니다.")

    cards = [(int(suit), int(rank)) for suit, rank in hand]

    ranks = [rank for suit, rank in cards]
    suits = [suit for suit, rank in cards]
    rank_counts = list(Counter(ranks).values())
    suit_counts = list(Counter(suits).values())
    rank_set = frozenset(ranks)
    suited = len(suit_counts) == 1

    if len(rank_set) == len(ranks):
        straight_windows = sum(1 for s in STRAIGHT_SETS if rank_set <= s)
    else:
        straight_windows = 0

    return [
        len(cards),
        max(rank_counts),
        rank_counts.count(2),
        rank_counts.count(3),
        len(rank_set),
        max(suit_counts),
        len(suit_counts),
        straight_windows,
        straight_windows if suited else 0,
        int(suited and rank_set <= ROYAL_RANKS),
        hand_category(cards),
    ]

# 3. 45개 특성 조합에 대한 확률 캐시 (RandomForest 중복 호출 방지)
_proba_cache = {}

# given parameters:
#   hand: list of tuples (suit, rank); length is 3 or 4
#   opposite_score: the score of opposite player
#   
# return:
#   True if you want to call,
#   False if you want to fold
def predict_call(hand, opposite_score):
    # 상대방 점수가 0점(노페어)이면 최소 무승부($0) 이상이 보장되므로 무조건 Call
    if opposite_score == 0:
        return True

    feat = tuple(extract_features(hand))

    if feat not in _proba_cache:
        X_df = pd.DataFrame([feat], columns=FEATURE_COLUMNS)
        _proba_cache[feat] = model.predict_proba(X_df)[0]

    probs = _proba_cache[feat]

    # 상대 점수와 비교하여 승리 확률(p_win) 및 패배 확률(p_lose) 계산
    p_win = sum(p for c, p in zip(classes, probs) if c > opposite_score)
    p_lose = sum(p for c, p in zip(classes, probs) if c < opposite_score)
    ev_showdown = 3.0 * p_win - 3.0 * p_lose

    if len(hand) == 3:
        # 3장째 Fold 손실(-1) 및 4번째 카드 확인 옵션 가치를 고려한 임계값
        return ev_showdown >= -1.2
    else:
        # 4장째 Fold 손실(-2) 대비 쇼다운 기댓값 손익분기점
        return ev_showdown >= -2.0
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