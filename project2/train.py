# -*- coding: utf-8 -*-
"""
랜덤 포레스트 학습 및 평가 → poker-hand-model.sav

- RandomForestClassifier 하나로 3장/4장 데이터를 함께 학습한다 (n_cards 특성으로 구분).
- class_weight는 쓰지 않는다 (predict_proba를 실제 확률로 쓰기 위해).
- 하이퍼파라미터는 학습 데이터 안에서 GroupKFold(groups=game_id, 3겹)로 비교하고,
  고른 설정으로 학습 데이터 전체를 다시 학습해 테스트 데이터로 확인한다.

실행: python train.py  (결과는 표준 출력, 약 20분)
"""

import itertools
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from build_dataset import TEST_CSV, TRAIN_CSV, build_dataset
from features import FEATURE_COLUMNS


MODEL_PATH = Path(__file__).resolve().parent / "poker-hand-model.sav"

CLASSES = list(range(10))
RANDOM_STATE = 42
N_SPLITS = 3

PARAM_GRID = {
    "n_estimators": [100, 300],
    "min_samples_leaf": [1, 20, 100],
    "max_depth": [None, 15],
}

# CV log loss가 최솟값과 이만큼 이내면 같은 성능으로 보고 트리 수가 적은 쪽을 고른다
TIE_TOL = 1e-4

# 학습 CSV의 0~9행은 무작위 패가 아니라 일부러 넣은 예시라 뺀다
# (0~4 로열 플러시, 5~9 스트레이트 플러시: 그대로 두면 로열 플러시 확률이 부풀려진다)
EXCLUDE_GAME_IDS = tuple(range(10))


def to_frame(X):
    return pd.DataFrame(X, columns=FEATURE_COLUMNS)


def aggregate(X, y):
    """샘플을 (특성 조합, target)별 개수로 묶는다. 특성 조합이 45개뿐이라 평가가 빨라진다."""
    df = to_frame(X)
    df["target"] = y
    return df.groupby(FEATURE_COLUMNS + ["target"]).size().rename("count").reset_index()


def evaluate(model, agg):
    """전체 / 3장 / 4장 log loss와 accuracy (개수를 sample_weight로 넘기면 샘플별 계산과 같다)."""
    P = np.zeros((len(agg), len(CLASSES)))
    P[:, model.classes_] = model.predict_proba(agg[FEATURE_COLUMNS])
    y = agg["target"].to_numpy()
    w = agg["count"].to_numpy()
    correct = P.argmax(axis=1) == y

    result = {}
    for label, m in (("전체", np.ones(len(agg), bool)),
                     ("3장", agg["n_cards"].to_numpy() == 3),
                     ("4장", agg["n_cards"].to_numpy() == 4)):
        result[f"logloss_{label}"] = log_loss(y[m], P[m], sample_weight=w[m], labels=CLASSES)
        result[f"acc_{label}"] = np.average(correct[m], weights=w[m])
    return result


def make_model(params):
    return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)


def cross_validate(X, y, groups):
    folds = GroupKFold(n_splits=N_SPLITS).split(X, y, groups)
    fold_data = [(to_frame(X[tr]), y[tr], aggregate(X[va], y[va])) for tr, va in folds]

    rows = []
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID, values))
        scores = [evaluate(make_model(params).fit(X_tr, y_tr), agg_va)
                  for X_tr, y_tr, agg_va in fold_data]
        row = {**params, **pd.DataFrame(scores).mean().to_dict(),
               "logloss_fold_std": np.std([s["logloss_전체"] for s in scores])}
        rows.append(row)
        print(f"  {params} → CV log loss {row['logloss_전체']:.5f}", flush=True)
    return pd.DataFrame(rows)


def choose(cv):
    best = cv["logloss_전체"].min()
    tied = cv[cv["logloss_전체"] <= best + TIE_TOL]
    choice = tied.sort_values(["n_estimators", "logloss_전체"]).iloc[0]
    return {name: (None if pd.isna(choice[name]) else int(choice[name])) for name in PARAM_GRID}


if __name__ == "__main__":

    start = time.perf_counter()

    X, y, groups = build_dataset(TRAIN_CSV)
    X_test, y_test, _ = build_dataset(TEST_CSV)

    keep = ~np.isin(groups, EXCLUDE_GAME_IDS)
    X, y, groups = X[keep], y[keep], groups[keep]
    print(f"학습: {len(X):,} 샘플 (게임 {len(np.unique(groups)):,}개, 제외 game_id {list(EXCLUDE_GAME_IDS)})")
    print(f"테스트: {len(X_test):,} 샘플")

    # ---- 하이퍼파라미터 비교
    print(f"\n하이퍼파라미터 비교 (GroupKFold {N_SPLITS}겹)", flush=True)
    cv = cross_validate(X, y, groups)
    params = choose(cv)

    shown = cv.assign(max_depth=cv["max_depth"].map(lambda v: "None" if pd.isna(v) else int(v)))
    print("\n=== 교차 검증 결과 (log loss 낮은 순) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(shown.sort_values("logloss_전체").to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\n선택한 설정: {params}")

    # ---- 최종 학습 + 테스트 평가
    model = make_model(params).fit(to_frame(X), y)
    result = evaluate(model, aggregate(X_test, y_test))
    print("\n=== 테스트 결과 ===")
    for label in ("전체", "3장", "4장"):
        print(f"{label}: log loss {result[f'logloss_{label}']:.5f}, accuracy {result[f'acc_{label}']:.4f}")

    # ---- 저장
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n저장: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024 ** 2:.2f} MB)")
    print("model.classes_:", model.classes_.tolist())
    print("\n=== feature_importances_ ===")
    print(pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
          .to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\n총 소요 시간: {(time.perf_counter() - start) / 60:.1f}분")
