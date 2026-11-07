# -*- coding: utf-8 -*-
"""
족보 확률 예측 모델용 특성 함수.

3장 또는 4장의 현재 패에서 최종 5장 족보를 정하는 "구조"만 특성으로 뽑는다.
문양 번호, 숫자 원본값, 카드 순서는 특성에 쓰지 않는다.
표준 라이브러리만 사용한다 (poker.py의 FILL HERE 블록에 그대로 붙여넣기 위해).
"""

from collections import Counter


# 스트레이트 조합 (A는 1 또는 14로 사용 가능)
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
