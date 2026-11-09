## -------------------------------------------------
from collections import Counter
import pickle
import numpy as np
import pandas as pd

# 1. features.py 내용 전체 (원본과 100% 동일)
STRAIGHT_SETS = [frozenset(range(low, low + 5)) for low in range(1, 10)]
STRAIGHT_SETS.append(frozenset((1, 10, 11, 12, 13)))

ROYAL_RANKS = frozenset((1, 10, 11, 12, 13))


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


def extract_features(hand):
    """(suit, rank) 튜플 3~4개 → FEATURE_COLUMNS 순서의 정수 리스트."""

    if len(hand) not in (3, 4):
        raise ValueError("현재 카드가 3장 또는 4장이어야 합니다.")

    cards = [(int(suit), int(rank)) for suit, rank in hand]

    for suit, rank in cards:
        if not (1 <= suit <= 4 and 1 <= rank <= 13):
            raise ValueError(f"카드 숫자 또는 문양 범위 오류: {(suit, rank)}")
    if len(set(cards)) != len(cards):
        raise ValueError("동일한 카드가 중복되었습니다.")

    ranks = [rank for suit, rank in cards]
    suits = [suit for suit, rank in cards]
    rank_counts = list(Counter(ranks).values())
    suit_counts = list(Counter(suits).values())
    rank_set = frozenset(ranks)
    suited = len(suit_counts) == 1

    # 숫자가 중복되면 스트레이트 불가 → 0
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


# 2. 모델 로드 및 확률 예측 함수 (가이드 2-3 준수 + 45개 조합 캐싱 최적화)
with open("poker-hand-model.sav", "rb") as f:
    hand_model = pickle.load(f)

_proba_cache = {}


def predict_hand_proba(hand):
    """현재 패(3~4장) → 최종 5장 족보 0~9 확률 (길이 10 numpy 배열, 합 1)."""
    feat = tuple(extract_features(hand))
    if feat not in _proba_cache:
        X = pd.DataFrame([feat], columns=FEATURE_COLUMNS)
        proba = np.zeros(10)
        proba[hand_model.classes_] = hand_model.predict_proba(X)[0]
        _proba_cache[feat] = proba
    return _proba_cache[feat]


def win_tie_proba(hand, opposite_score):
    """(최종 족보 > 상대 점수일 확률, 최종 족보 == 상대 점수일 확률)."""
    proba = predict_hand_proba(hand)
    return proba[opposite_score + 1:].sum(), proba[opposite_score]


# 3. Call / Fold 베팅 의사결정 함수 (조원 구현부)
def predict_call(hand, opposite_score):
    # 상대방이 0점(노페어)이면 최소 무승부 이상이므로 무조건 Call
    if opposite_score == 0:
        return True

    p_win, p_tie = win_tie_proba(hand, opposite_score)

    # 쇼다운 시 돌려받는 기대 상금: 승리 시 +$6, 무승부 시 +$3
    expected_return = 6.0 * p_win + 3.0 * p_tie

    if len(hand) == 3:
        # 3장 단계: 끝까지 갈 때 추가 비용 $2 대비 4장 확인 후 중도 포기(Fold) 옵션 가치를 반영한 임계값 ($1.8)
        # (순수익 기준 ev_showdown >= -1.2 와 수학적으로 완전히 동일한 식)
        return expected_return >= 1.8
    else:
        # 4장 단계: 마지막 1장 추가 베팅 비용 $1 이상이면 Call
        # (가이드 2-5의 6*p_win + 3*p_tie - 1 >= 0 및 ev_showdown >= -2.0 과 동일)
        return expected_return >= 1.0
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