import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import ttest_rel

from xgboost import XGBClassifier


# ------------------------------------------------------------
# 1. 기본 설정
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "week2_results" / "dataset.npz"
OUTPUT_DIR = PROJECT_DIR / "week3_results"

RANDOM_STATE = 42
N_SPLITS = 5

# 2초 구간, 200ms 창, 100ms hop이면 파일당 19개 윈도우
WINDOWS_PER_FILE = 19

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. 데이터 불러오기 및 원본 특징값 복원
# ------------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"데이터셋을 찾을 수 없습니다: {DATA_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:

        required = [
            "X_train", "y_train",
            "X_test", "y_test",
            "scaler_mean", "scaler_scale",
        ]

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"dataset.npz에 {key} 항목이 없습니다."
                )

        X_train = np.asarray(data["X_train"], dtype=np.float64)
        y_train = np.asarray(data["y_train"]).astype(str)

        X_test = np.asarray(data["X_test"], dtype=np.float64)
        y_test = np.asarray(data["y_test"]).astype(str)

        scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(data["scaler_scale"], dtype=np.float64)

    # 저장된 X_train이 이미 표준화된 값인지 확인
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)

    looks_standardized = (
        np.all(np.abs(means) < 0.05)
        and np.all(
            (np.abs(stds - 1.0) < 0.15)
            | (stds < 0.05)
        )
    )

    if looks_standardized:

        # 기존 표준화를 되돌린 뒤
        # 각 CV 폴드에서 표준화를 새로 학습한다.
        X_train = X_train * scaler_scale + scaler_mean
        X_test = X_test * scaler_scale + scaler_mean

        print("기존 표준화가 적용된 특징값을 복원했습니다.")

    else:
        print("저장된 특징값을 원본 특징값으로 사용합니다.")

    # CV는 250개 파일 전체를 대상으로 수행한다.
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    if len(X) % WINDOWS_PER_FILE != 0:
        raise ValueError(
            "전체 샘플 수가 파일당 윈도우 개수로 나누어지지 않습니다."
        )

    n_files = len(X) // WINDOWS_PER_FILE

    print(f"\n전체 윈도우 수: {len(X)}")
    print(f"파일당 윈도우 수: {WINDOWS_PER_FILE}")
    print(f"추정 파일 수: {n_files}")

    # 파일 하나에서 나온 19개 윈도우의 라벨이 같은지 확인
    y_by_file = y.reshape(n_files, WINDOWS_PER_FILE)

    same_label = np.all(
        y_by_file == y_by_file[:, :1],
        axis=1,
    )

    if not np.all(same_label):
        raise ValueError(
            "연속된 19개 윈도우의 라벨이 일치하지 않습니다.\n"
            "파일별로 데이터가 저장된 순서를 확인해야 합니다. "
            "잘못된 분할로 CV를 진행하지 않고 중단합니다."
        )

    # 각 파일을 대표하는 라벨
    file_labels = y_by_file[:, 0]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"클래스: {encoder.classes_.tolist()}")
    print("파일별 라벨 확인 완료")

    return X, y_encoded, file_labels, encoder.classes_, n_files


# ------------------------------------------------------------
# 3. 모델 설정
# ------------------------------------------------------------

def create_models(n_classes):

    return {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            decision_function_shape="ovr",
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
        ),
    }


# ------------------------------------------------------------
# 4. 파일 단위 5-fold 교차검증
# ------------------------------------------------------------

def run_cross_validation(X, y, file_labels, n_files, models):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    # 파일 인덱스를 해당 파일의 윈도우 인덱스로 확장
    def expand_indices(file_indices):

        return np.concatenate([
            np.arange(
                file_index * WINDOWS_PER_FILE,
                (file_index + 1) * WINDOWS_PER_FILE,
            )
            for file_index in file_indices
        ]).astype(int)

    for fold, (train_files, valid_files) in enumerate(
        skf.split(np.arange(n_files), file_labels),
        start=1,
    ):

        train_idx = expand_indices(train_files)
        valid_idx = expand_indices(valid_files)

        # 파일이 학습/검증 양쪽에 동시에 들어가는지 확인
        overlap = set(train_files) & set(valid_files)

        if overlap:
            raise RuntimeError(
                f"{fold}번 폴드에서 파일 누수가 발견되었습니다."
            )

        print("\n" + "=" * 65)
        print(f"{fold}/{N_SPLITS}번 폴드")
        print(f"학습 파일: {len(train_files)}개")
        print(f"검증 파일: {len(valid_files)}개")
        print(f"학습 윈도우: {len(train_idx)}개")
        print(f"검증 윈도우: {len(valid_idx)}개")
        print("=" * 65)

        for model_name, prototype in models.items():

            # 각 폴드마다 새 모델과 scaler 사용
            # scaler는 해당 폴드의 학습 데이터에만 fit
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", clone(prototype)),
            ])

            start_time = time.perf_counter()

            pipeline.fit(X[train_idx], y[train_idx])

            train_time = time.perf_counter() - start_time

            predictions = pipeline.predict(X[valid_idx])

            acc = accuracy_score(
                y[valid_idx], predictions
            )

            precision = precision_score(
                y[valid_idx],
                predictions,
                average="macro",
                zero_division=0,
            )

            recall = recall_score(
                y[valid_idx],
                predictions,
                average="macro",
                zero_division=0,
            )

            f1 = f1_score(
                y[valid_idx],
                predictions,
                average="macro",
                zero_division=0,
            )

            row = {
                "Fold": fold,
                "Model": model_name,
                "Accuracy": acc,
                "Precision_macro": precision,
                "Recall_macro": recall,
                "F1_macro": f1,
                "Train_time_sec": train_time,
            }

            fold_rows.append(row)

            print(
                f"{model_name:16s} | "
                f"Accuracy: {acc:.4f} | "
                f"Macro F1: {f1:.4f} | "
                f"학습: {train_time:.3f}초"
            )

    return fold_rows


# ------------------------------------------------------------
# 5. 평균 및 표준편차 계산
# ------------------------------------------------------------

def summarize_results(fold_rows, model_names):

    summary_rows = []

    print("\n" + "=" * 65)
    print("5-FOLD 교차검증 결과")
    print("=" * 65)

    for model_name in model_names:

        rows = [
            row for row in fold_rows
            if row["Model"] == model_name
        ]

        accuracies = np.array([
            row["Accuracy"] for row in rows
        ])

        precisions = np.array([
            row["Precision_macro"] for row in rows
        ])

        recalls = np.array([
            row["Recall_macro"] for row in rows
        ])

        f1_scores = np.array([
            row["F1_macro"] for row in rows
        ])

        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies, ddof=0)

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores, ddof=0)

        summary_rows.append({
            "Model": model_name,
            "Accuracy_mean": accuracy_mean,
            "Accuracy_std": accuracy_std,
            "Precision_mean": np.mean(precisions),
            "Recall_mean": np.mean(recalls),
            "F1_mean": f1_mean,
            "F1_std": f1_std,
        })

        fold_values = ", ".join(
            f"{value * 100:.2f}%"
            for value in accuracies
        )

        print(f"\n{model_name}")
        print(f"각 폴드 정확도: {fold_values}")
        print(
            f"Accuracy: "
            f"{accuracy_mean * 100:.2f} ± "
            f"{accuracy_std * 100:.2f}%"
        )
        print(
            f"Macro F1: "
            f"{f1_mean:.4f} ± {f1_std:.4f}"
        )

    return summary_rows


# ------------------------------------------------------------
# 6. 대응표본 t-검정
# ------------------------------------------------------------

def run_statistical_tests(fold_rows, model_names):

    accuracy_by_model = {
        name: np.array([
            row["Accuracy"]
            for row in fold_rows
            if row["Model"] == name
        ])
        for name in model_names
    }

    test_rows = []
    analysis_lines = []

    print("\n" + "=" * 65)
    print("대응표본 t-검정 결과")
    print("=" * 65)

    for model_a, model_b in combinations(model_names, 2):

        a = accuracy_by_model[model_a]
        b = accuracy_by_model[model_b]

        statistic, p_value = ttest_rel(a, b)

        if not np.isfinite(p_value):

            verdict = "검정 불가"

            interpretation = (
                "p값이 유효하지 않아 유의성 판단을 할 수 없습니다. "
                "폴드별 정확도와 차이의 분산을 확인해야 합니다."
            )

        elif p_value < 0.05:

            verdict = "유의함"

            interpretation = (
                "유의수준 0.05에서 두 모델의 폴드별 정확도 "
                "차이가 통계적으로 유의하게 나타났습니다."
            )

        else:

            verdict = "유의한 차이를 검출하지 못함"

            interpretation = (
                "유의수준 0.05에서 두 모델 간 정확도 차이를 "
                "통계적으로 검출하지 못했습니다. "
                "이는 두 모델이 동등하다는 증거는 아닙니다."
            )

        row = {
            "Model_A": model_a,
            "Model_B": model_b,
            "t_statistic": statistic,
            "p_value": p_value,
            "Verdict": verdict,
        }

        test_rows.append(row)

        line = (
            f"{model_a} vs {model_b}\n"
            f"t = {statistic:.4f}, p = {p_value:.6f}\n"
            f"판정: {verdict}\n"
            f"해석: {interpretation}\n"
        )

        analysis_lines.append(line)

        print(line)


    return test_rows, analysis_lines


# ------------------------------------------------------------
# 7. 결과 파일 저장
# ------------------------------------------------------------

def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    X, y, file_labels, class_names, n_files = load_data()

    models = create_models(len(class_names))
    model_names = list(models.keys())

    fold_rows = run_cross_validation(
        X,
        y,
        file_labels,
        n_files,
        models,
    )

    summary_rows = summarize_results(
        fold_rows,
        model_names,
    )

    test_rows, analysis_lines = run_statistical_tests(
        fold_rows,
        model_names,
    )

    # 폴드별 결과
    save_csv(
        OUTPUT_DIR / "cv_fold_results.csv",
        fold_rows,
    )

    # 평균 및 표준편차
    save_csv(
        OUTPUT_DIR / "cv_summary.csv",
        summary_rows,
    )

    # 모델 간 통계 검정 결과
    save_csv(
        OUTPUT_DIR / "paired_ttests.csv",
        test_rows,
    )

    # 통계 검정 해석
    (
        OUTPUT_DIR / "statistical_analysis.txt"
    ).write_text(
        "\n".join(analysis_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("5-fold 교차검증 및 통계 검정 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 65)

    print("생성된 파일:")
    print("- cv_fold_results.csv")
    print("- cv_summary.csv")
    print("- paired_ttests.csv")
    print("- statistical_analysis.txt")


if __name__ == "__main__":
    main()
