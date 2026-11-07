# -*- coding: utf-8 -*-
"""
학습 데이터 생성: 원본 CSV 각 행(5장)에서 3장 조합 10개 + 4장 조합 5개를 만든다.

반환: X (원본 행 수 × 15, 11) int8, y (최종 족보 0~9) int8, groups (game_id = 원본 행 번호) int32
결과는 poker_dataset_train.npz / poker_dataset_test.npz에 저장해 두고 다음부터 불러온다.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
from pandas import read_csv, to_numeric

from features import FEATURE_COLUMNS, extract_features, hand_category


FOLDER = Path(__file__).resolve().parent

TRAIN_CSV = "poker-hand-training.csv"
TEST_CSV = "poker-hand-testing.csv"

SUIT_COLS = [f"Suit of Card {n}" for n in range(1, 6)]
RANK_COLS = [f"Rank of Card {n}" for n in range(1, 6)]
TARGET_COL = "Poker Hand"

SAMPLES_PER_GAME = 10 + 5   # C(5,3) + C(5,4)


def validate_data(df):
    """열 존재, 값 범위, 카드 중복, hand_category와 정답 일치를 검사한다."""

    required = SUIT_COLS + RANK_COLS + [TARGET_COL]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {missing}")
    if df.empty:
        raise ValueError("CSV에 데이터가 없습니다.")

    for name in required:
        values = to_numeric(df[name], errors="raise").to_numpy()
        low, high = (1, 4) if name.startswith("Suit") else (1, 13) if name.startswith("Rank") else (0, 9)
        bad = np.isnan(values) | (values != np.round(values)) | (values < low) | (values > high)
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            raise ValueError(f"{name} 열에 잘못된 값이 있습니다: {row + 2}행 값={values[row]}")

    suits = df[SUIT_COLS].to_numpy(dtype=np.int64)
    ranks = df[RANK_COLS].to_numpy(dtype=np.int64)
    target = df[TARGET_COL].to_numpy(dtype=np.int64)

    # 게임당 동일 카드 중복 검사
    codes = np.sort((suits - 1) * 13 + (ranks - 1), axis=1)
    dup = (np.diff(codes, axis=1) == 0).any(axis=1)
    if dup.any():
        raise ValueError(f"{int(np.flatnonzero(dup)[0]) + 2}행 카드 중복")

    # 최종 족보 정답 검증
    for game_id, (s_row, r_row) in enumerate(zip(suits.tolist(), ranks.tolist())):
        computed = hand_category(list(zip(s_row, r_row)))
        if computed != target[game_id]:
            raise ValueError(f"{game_id + 2}행 족보 불일치: 정답={target[game_id]}, 계산={computed}")

    return suits, ranks, target


def _compute(csv_path):

    suits, ranks, target = validate_data(read_csv(csv_path))
    n_games = len(target)

    # 캐시: 정렬된 카드 튜플 → unique_rows 안의 위치
    cache, unique_rows = {}, []
    row_index = np.empty(n_games * SAMPLES_PER_GAME, dtype=np.int32)

    k = 0
    for s_row, r_row in zip(suits.tolist(), ranks.tolist()):
        cards = tuple(sorted(zip(s_row, r_row)))
        for size in (3, 4):
            for combo in combinations(cards, size):
                idx = cache.get(combo)
                if idx is None:
                    idx = cache[combo] = len(unique_rows)
                    unique_rows.append(extract_features(combo))
                row_index[k] = idx
                k += 1

    X = np.asarray(unique_rows, dtype=np.int8)[row_index]
    y = np.repeat(target, SAMPLES_PER_GAME).astype(np.int8)
    groups = np.repeat(np.arange(n_games, dtype=np.int32), SAMPLES_PER_GAME)
    return X, y, groups


def build_dataset(csv_path, use_cache=True):
    """CSV → (X, y, groups). use_cache=True이고 npz가 있으면 다시 계산하지 않는다."""

    csv_path = FOLDER / csv_path
    kind = "train" if "train" in csv_path.name else "test"
    npz_path = FOLDER / f"poker_dataset_{kind}.npz"

    if use_cache and npz_path.exists():
        with np.load(npz_path) as data:
            if list(data["feature_columns"]) == FEATURE_COLUMNS:
                return data["X"], data["y"], data["groups"]

    X, y, groups = _compute(csv_path)
    np.savez_compressed(npz_path, X=X, y=y, groups=groups,
                        feature_columns=np.array(FEATURE_COLUMNS))
    return X, y, groups
