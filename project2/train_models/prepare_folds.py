# prepare_folds.py

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import KFold


#파일 선정

INPUT_CSV = "poker-hand-testing.csv"

# ==========================================
# 1. 경로 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_DIR))

from feature_engine import (
    INPUT_CSV,
    build_features
)

FOLD_DIR = BASE_DIR / "folds"
FOLD_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. 원본 데이터 불러오기
# ==========================================

df = pd.read_csv(
    PROJECT_DIR / INPUT_CSV
)

print("전체 게임 수:", len(df))


# ==========================================
# 3. 원본 데이터 5개 분할
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 4. 각 데이터 피처 엔지니어링
# ==========================================

for fold, (_, test_idx) in enumerate(
    kf.split(df),
    start=1
):

    print(f"\n=== 데이터 {fold} 처리 ===")

    # 원본 데이터 중 해당 그룹 추출
    fold_data = df.iloc[test_idx].copy()

    fold_data = fold_data.reset_index(
        drop=True
    )

    # 원본 분할 데이터 저장
    fold_data.to_csv(
        FOLD_DIR / f"raw_fold_{fold}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 기존 피처 엔지니어링 함수 사용
    feature_data = build_features(fold_data)

    # 피처 엔지니어링 결과 저장
    feature_data.to_csv(
        FOLD_DIR / f"features_fold_{fold}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("원본 게임:", len(fold_data))
    print("피처 데이터:", len(feature_data))


print("\n5개 데이터 피처 엔지니어링 완료")