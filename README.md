# 2026-understanding-ai

## 정확도(Accuracy)와 F1 점수 비교

| 구분 | 정확도 (Accuracy) | F1 점수 (F1-Score) |
|---|---|---|
| 중점 요소 | 전체 정답률 (맞춘 개수 / 전체 개수) | 정밀도와 재현율의 균형 (조화 평균) |
| 공식 | $\dfrac{TP + TN}{\text{전체 데이터 수}}$ | $2 \times \dfrac{Precision \times Recall}{Precision + Recall}$<br>Precision $= \dfrac{TP}{TP + FP}$ (True라고 예측한 것 중 실제 True)<br>Recall $= \dfrac{TP}{TP + FN}$ (실제 True 중 True라고 맞춘 것) |
| 데이터 불균형 취약성 | **매우 취약함** (데이터 비율이 한쪽으로 치우치면 왜곡 발생) | **강함** (소수 클래스의 성능 저하를 엄격하게 반영함) |
| 값의 범위 | 0 ~ 1 (0% ~ 100%) | 0 ~ 1 (높을수록 좋음) |
| 주요 사용 사례 | 클래스 비율이 **균형**을 이룰 때 | 의료 진단, 사기 탐지 등 **불균형 데이터**일 때 |
| 실제 결과: Decision Tree | 0.658552 | 0.221992 (최고) |
| 실제 결과: Random Forest | 0.685239 (최고) | 0.216410 |
| 실제 결과: Extra Trees | 0.684213 | 0.216004 |

- 실제 결과: `project2/train_models` (`First_step` 브랜치)에서 `poker-hand-testing.csv`를 5-Fold 교차 검증한 평균값. F1은 족보 0~9 클래스의 Macro F1.
- 정확도는 Random Forest가, Macro F1은 Decision Tree가 가장 높다. 족보 분포가 한쪽(낮은 족보)에 크게 치우쳐 있어 정확도는 약 0.66~0.69지만, 드문 족보를 잘 맞추지 못해 Macro F1은 약 0.22에 그친다.
