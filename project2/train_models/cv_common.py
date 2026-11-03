# cv_common.py

from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score
)


BASE_DIR = Path(__file__).resolve().parent

FOLD_DIR = BASE_DIR / "folds"


def run_cross_validation(model, model_name, model_number):

    # ==========================================
    # 1. 5개 데이터 불러오기
    # ==========================================

    datasets = []

    for i in range(1, 6):

        df = pd.read_csv(
            FOLD_DIR / f"features_fold_{i}.csv"
        )

        datasets.append(df)

    # ==========================================
    # 2. 피처 설정
    # ==========================================

    exclude_columns = [
        "game_id",
        "target_poker_hand"
    ]

    feature_columns = [
        col for col in datasets[0].columns
        if col not in exclude_columns
        and not col.startswith("Unnamed:")
    ]

    if len(feature_columns) != 20:
        raise ValueError(
            f"입력 피처 수 오류: {len(feature_columns)}"
        )

    results = []

    # ==========================================
    # 3. 5회 교차 검증
    # ==========================================

    for fold in range(5):

        print(f"\n=== {model_name}: {fold + 1}회차 ===")

        # 평가 데이터 1개
        test_data = datasets[fold]

        # 학습 데이터 4개
        train_data = pd.concat(
            [
                datasets[i]
                for i in range(5)
                if i != fold
            ],
            ignore_index=True
        )

        X_train = train_data[feature_columns]
        y_train = train_data["target_poker_hand"]

        X_test = test_data[feature_columns]
        y_test = test_data["target_poker_hand"]

        # 새로운 모델 생성
        current_model = clone(model)

        # 모델 학습
        current_model.fit(
            X_train,
            y_train
        )

        # 예측
        predictions = current_model.predict(
            X_test
        )

        # 정확도 계산
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # Macro F1 계산
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=list(range(10)),
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Fold": fold + 1,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1
        })

        print("학습 데이터:", len(X_train))
        print("평가 데이터:", len(X_test))
        print(f"정확도: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    # ==========================================
    # 4. 5회 결과 평균
    # ==========================================

    result_df = pd.DataFrame(results)

    accuracy_mean = result_df["Accuracy"].mean()
    f1_mean = result_df["Macro_F1"].mean()

    print(f"\n=== {model_name} 최종 결과 ===")
    print(f"평균 정확도: {accuracy_mean:.4f}")
    print(f"평균 Macro F1: {f1_mean:.4f}")

    # ==========================================
    # 5. 결과 저장
    # ==========================================

    result_df.to_csv(
        BASE_DIR / f"fold_results_{model_number}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = pd.DataFrame([{
        "Model": model_name,
        "Accuracy_Mean": accuracy_mean,
        "Macro_F1_Mean": f1_mean
    }])

    summary.to_csv(
        BASE_DIR / f"summary_{model_number}.csv",
        index=False,
        encoding="utf-8-sig"
    )