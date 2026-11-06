# compare_models.py

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. 세 모델 결과 불러오기
# ==========================================

results = []

for i in range(1, 4):

    df = pd.read_csv(
        BASE_DIR / f"summary_{i}.csv"
    )

    results.append(df)


# ==========================================
# 2. 결과 통합
# ==========================================

comparison = pd.concat(
    results,
    ignore_index=True
)


# ==========================================
# 3. 평균 정확도 비교
# ==========================================

accuracy_ranking = comparison.sort_values(
    by="Accuracy_Mean",
    ascending=False
)

print("\n=== 평균 정확도 순위 ===")

print(
    accuracy_ranking.to_string(index=False)
)


# ==========================================
# 4. 평균 Macro F1 비교
# ==========================================

f1_ranking = comparison.sort_values(
    by="Macro_F1_Mean",
    ascending=False
)

print("\n=== 평균 Macro F1 순위 ===")

print(
    f1_ranking.to_string(index=False)
)


# ==========================================
# 5. 최고 성능 모델 출력
# ==========================================

print("\n=== 최종 비교 결과 ===")

print(
    "정확도 최고 모델:",
    accuracy_ranking.iloc[0]["Model"]
)

print(
    "Macro F1 최고 모델:",
    f1_ranking.iloc[0]["Model"]
)


# ==========================================
# 6. 결과 저장
# ==========================================

comparison.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n모델 비교 완료!")