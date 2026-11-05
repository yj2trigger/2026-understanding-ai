from sklearn.ensemble import RandomForestClassifier
from cv_common import run_cross_validation

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=40,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

run_cross_validation(
    model,
    "Random Forest",
    2
)