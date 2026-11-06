# -*- coding: utf-8 -*-

from collections import Counter
from pathlib import Path
from pandas import DataFrame, read_csv, to_numeric


# ==================================================
# 1. CSV 파일 설정
# ==================================================

INPUT_CSV = "poker-hand-training.csv"

OUTPUT_3_CSV = "poker_features_3cards.csv"
OUTPUT_4_CSV = "poker_features_4cards.csv"
OUTPUT_ALL_CSV = "poker_features_all.csv"
EVENT_LOG_CSV = "poker_event_log.csv"


# ==================================================
# 2. 원본 데이터 열 설정
# ==================================================

SUIT_COLS = [
    f"Suit of Card {n}"
    for n in range(1, 6)
]

RANK_COLS = [
    f"Rank of Card {n}"
    for n in range(1, 6)
]

CARD_COLS = [
    name
    for n in range(5)
    for name in (SUIT_COLS[n], RANK_COLS[n])
]


# ==================================================
# 3. 포커 족보 점수
# ==================================================

HAND_NAMES = (
    "노페어",
    "원페어",
    "투페어",
    "트리플",
    "스트레이트",
    "플러시",
    "풀하우스",
    "포카드",
    "스트레이트 플러시",
    "로열 플러시"
)


# 스트레이트 조합
# A는 1 또는 14로 사용 가능

STRAIGHT_SETS = [
    frozenset(range(low, low + 5))
    for low in range(1, 10)
]

STRAIGHT_SETS.append(
    frozenset((1, 10, 11, 12, 13))
)


# ==================================================
# 4. 모델 입력 피처 설정
# ==================================================

FEATURE_COLUMNS = ["stage_cards"]

for n in range(1, 5):

    FEATURE_COLUMNS.extend((
        f"rank_{n}",
        f"suit_{n}"
    ))

FEATURE_COLUMNS.extend((
    "rank_unique",
    "rank_max_count",
    "pair_count",
    "triple_count",
    "suit_max_count",
    "longest_straight_run",
    "straight_possible",
    "flush_possible",
    "straight_flush_possible",
    "current_hand_score",
    "event_bonus"
))


# ==================================================
# 5. 현재 카드의 족보 점수 계산
# ==================================================

def hand_category(hand):

    if len(hand) not in (3, 4, 5):
        raise ValueError(
            "카드는 3장, 4장 또는 5장이어야 합니다."
        )

    ranks = [
        rank
        for suit, rank in hand
    ]

    suits = [
        suit
        for suit, rank in hand
    ]

    frequencies = sorted(
        Counter(ranks).values(),
        reverse=True
    )

    # 5장일 때 완성된 족보 검사
    if len(hand) == 5:

        rank_set = frozenset(ranks)

        straight = (
            len(rank_set) == 5
            and rank_set in STRAIGHT_SETS
        )

        flush = len(set(suits)) == 1

        if straight and flush:

            if rank_set == frozenset(
                (1, 10, 11, 12, 13)
            ):
                return 9

            return 8

        if 4 in frequencies:
            return 7

        if frequencies == [3, 2]:
            return 6

        if flush:
            return 5

        if straight:
            return 4

    # 3~5장에서 숫자 조합 검사

    if 4 in frequencies:
        return 7

    if 3 in frequencies:
        return 3

    if frequencies.count(2) == 2:
        return 2

    if 2 in frequencies:
        return 1

    return 0


# ==================================================
# 6. 연속된 카드 숫자 분석
# ==================================================

def longest_straight_run(ranks):

    values = set(ranks)

    # A를 14로도 취급
    if 1 in values:
        values.add(14)

    ordered = sorted(values)

    longest = 1
    current = 1

    for before, after in zip(
        ordered,
        ordered[1:]
    ):

        if after == before + 1:
            current += 1

        else:
            current = 1

        longest = max(
            longest,
            current
        )

    return longest


# ==================================================
# 7. 스트레이트 완성 가능성
# ==================================================

def straight_possible(ranks):

    rank_set = frozenset(ranks)

    possible = (
        len(rank_set) == len(ranks)
        and any(
            rank_set.issubset(straight)
            for straight in STRAIGHT_SETS
        )
    )

    return int(possible)


# ==================================================
# 8. 피처 엔지니어링 핵심 함수
# ==================================================

def features_for_hand(hand):

    if len(hand) not in (3, 4):

        raise ValueError(
            "현재 카드가 3장 또는 4장이어야 합니다."
        )

    cards = []

    for suit, rank in hand:

        suit = int(suit)
        rank = int(rank)

        if not (
            1 <= suit <= 4
            and 1 <= rank <= 13
        ):

            raise ValueError(
                "카드 숫자 또는 무늬 범위 오류"
            )

        cards.append((suit, rank))

    # 같은 카드 중복 검사
    if len(set(cards)) != len(cards):

        raise ValueError(
            "동일한 카드가 중복되었습니다."
        )

    # 카드 정렬
    # 숫자와 무늬를 하나의 쌍으로 유지

    ordered_cards = sorted(
        (rank, suit)
        for suit, rank in cards
    )

    ranks = [
        rank
        for rank, suit in ordered_cards
    ]

    suits = [
        suit
        for rank, suit in ordered_cards
    ]

    frequencies = list(
        Counter(ranks).values()
    )

    # --------------------------------
    # 현재 족보 점수
    # --------------------------------

    current_score = hand_category(cards)

    # --------------------------------
    # 이벤트 추가 점수
    # --------------------------------

    if len(cards) == 4:

        previous_score = hand_category(
            cards[:3]
        )

    else:
        previous_score = 0

    event_bonus = max(
        0,
        current_score - previous_score
    )

    # --------------------------------
    # 원본 카드 피처 생성
    # --------------------------------

    result = {
        "stage_cards": len(cards)
    }

    for n in range(4):

        if n < len(cards):

            result[f"rank_{n + 1}"] = (
                ordered_cards[n][0]
            )

            result[f"suit_{n + 1}"] = (
                ordered_cards[n][1]
            )

        else:

            result[f"rank_{n + 1}"] = 0
            result[f"suit_{n + 1}"] = 0

    # --------------------------------
    # 족보 조합 피처 생성
    # --------------------------------

    result.update({

        # 서로 다른 숫자의 개수
        "rank_unique": len(set(ranks)),

        # 동일 숫자의 최대 등장 횟수
        "rank_max_count": max(frequencies),

        # 페어 개수
        "pair_count": frequencies.count(2),

        # 트리플 개수
        "triple_count": frequencies.count(3),

        # 같은 무늬의 최대 개수
        "suit_max_count": max(
            Counter(suits).values()
        ),

        # 가장 긴 연속 숫자
        "longest_straight_run":
            longest_straight_run(ranks),

        # 스트레이트 가능 여부
        "straight_possible":
            straight_possible(ranks),

        # 플러시 가능 여부
        "flush_possible": int(
            len(set(suits)) == 1
        ),

        # 스트레이트 플러시 가능 여부
        "straight_flush_possible": int(
            len(set(suits)) == 1
            and straight_possible(ranks) == 1
        ),

        # 현재 족보 점수
        "current_hand_score":
            current_score,

        # 추가 카드로 상승한 점수
        "event_bonus":
            event_bonus

    })

    return {
        col: result[col]
        for col in FEATURE_COLUMNS
    }


# ==================================================
# 9. 원본 데이터 검증
# ==================================================

def validate_data(df):

    required_cols = (
        CARD_COLS + ["Poker Hand"]
    )

    missing = [
        name
        for name in required_cols
        if name not in df.columns
    ]

    if missing:

        raise ValueError(
            f"필요한 열이 없습니다: {missing}"
        )

    if df.empty:

        raise ValueError(
            "CSV에 데이터가 없습니다."
        )

    clean = df.copy()

    for name in required_cols:

        clean[name] = to_numeric(
            clean[name],
            errors="raise"
        )

        if clean[name].isna().any():

            raise ValueError(
                f"{name} 열에 결측치가 있습니다."
            )

        if name.startswith("Suit"):

            valid = range(1, 5)

        elif name.startswith("Rank"):

            valid = range(1, 14)

        else:

            valid = range(10)

        if not clean[name].isin(valid).all():

            raise ValueError(
                f"{name} 열에 잘못된 값이 있습니다."
            )

        clean[name] = clean[name].astype(int)

    # 게임당 동일 카드 중복 검사
    # 최종 족보 정답도 검증

    for game_id, vals in enumerate(
        clean[CARD_COLS].itertuples(
            index=False,
            name=None
        )
    ):

        cards = [
            (
                int(vals[i]),
                int(vals[i + 1])
            )
            for i in range(0, 10, 2)
        ]

        if len(set(cards)) != 5:

            raise ValueError(
                f"{game_id + 2}행 카드 중복"
            )

        computed = hand_category(cards)

        expected = int(
            clean.iloc[game_id]["Poker Hand"]
        )

        if computed != expected:

            raise ValueError(
                f"{game_id + 2}행 족보 불일치: "
                f"정답={expected}, 계산={computed}"
            )

    return clean


# ==================================================
# 10. 전체 데이터 피처 엔지니어링
# ==================================================

def build_features(df):

    clean = validate_data(df)

    records = []

    for game_id, vals in enumerate(
        clean[CARD_COLS].itertuples(
            index=False,
            name=None
        )
    ):

        cards = [
            (
                int(vals[i]),
                int(vals[i + 1])
            )
            for i in range(0, 10, 2)
        ]

        target = int(
            clean.iloc[game_id]["Poker Hand"]
        )

        # 3장과 4장 상태 분리
        for stage_cards in (3, 4):

            record = {
                "game_id": game_id
            }

            record.update(
                features_for_hand(
                    cards[:stage_cards]
                )
            )

            # 최종 족보 정답
            record["target_poker_hand"] = target

            records.append(record)

    return DataFrame(
        records,
        columns=[
            "game_id",
            *FEATURE_COLUMNS,
            "target_poker_hand"
        ]
    )


# ==================================================
# 11. 이벤트 점수 기록
# ==================================================

def build_event_log(df):

    clean = validate_data(df)

    records = []

    for game_id, vals in enumerate(
        clean[CARD_COLS].itertuples(
            index=False,
            name=None
        )
    ):

        cards = [
            (
                int(vals[i]),
                int(vals[i + 1])
            )
            for i in range(0, 10, 2)
        ]

        before = 0

        for stage_cards in (3, 4, 5):

            score = hand_category(
                cards[:stage_cards]
            )

            records.append({

                "game_id": game_id,

                "stage_cards": stage_cards,

                "hand_name": HAND_NAMES[score],

                "current_hand_score": score,

                "event_bonus": max(
                    0,
                    score - before
                )

            })

            before = score

    return DataFrame(records)


# ==================================================
# 12. 실행
# ==================================================

if __name__ == "__main__":

    folder = Path(__file__).resolve().parent

    # CSV 데이터를 df로 불러오기
    df = read_csv(folder / INPUT_CSV)

    # 전처리 및 피처 엔지니어링
    df_features = build_features(df)

    # 3장 데이터
    df_features_3 = df_features[
        df_features["stage_cards"] == 3
    ].copy()

    # 4장 데이터
    df_features_4 = df_features[
        df_features["stage_cards"] == 4
    ].copy()

    # 이벤트 기록
    event_log = build_event_log(df)

    # --------------------------------
    # CSV 저장
    # --------------------------------

    df_features.to_csv(
        folder / OUTPUT_ALL_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    df_features_3.to_csv(
        folder / OUTPUT_3_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    df_features_4.to_csv(
        folder / OUTPUT_4_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    event_log.to_csv(
        folder / EVENT_LOG_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------
    # 결과 확인
    # --------------------------------

    print("원본 게임 수:", len(df))

    print("3장 데이터:", len(df_features_3))

    print("4장 데이터:", len(df_features_4))

    print("입력 피처 수:", len(FEATURE_COLUMNS))

    print(
        df_features[
            [
                "game_id",
                "stage_cards",
                "current_hand_score",
                "event_bonus",
                "target_poker_hand"
            ]
        ].head(10)
    )

    print("피처 엔지니어링 완료")
    print("\n=== 정답 분포 ===")
if __name__ == "__main__":

    # 기존 데이터 전처리 및 CSV 저장 코드
    # ...

    print("\n=== 정답 분포 ===")
    print(df_features["target_poker_hand"].value_counts().sort_index())

    print("\n=== 이벤트 보너스 분포 ===")
    print(df_features["event_bonus"].value_counts())