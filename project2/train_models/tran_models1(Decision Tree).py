from sklearn.tree import DecisionTreeClassifier
from cv_common import run_cross_validation

model = DecisionTreeClassifier(
    max_depth=16,
    min_samples_leaf=5,
    random_state=42
)

run_cross_validation(
    model,
    "Decision Tree",
    1
)