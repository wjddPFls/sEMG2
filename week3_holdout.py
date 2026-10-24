import sys
import time
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ------------------------------------------------------------
# 1. 경로 및 기본 설정
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "week2_results" / "dataset.npz"
OUTPUT_DIR = PROJECT_DIR / "week3_results"

RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. 데이터 불러오기
# ------------------------------------------------------------

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"데이터셋을 찾을 수 없습니다.\n"
            f"확인할 경로: {DATA_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:

        required_keys = [
            "X_train",
            "y_train",
            "X_test",
            "y_test",
        ]

        for key in required_keys:
            if key not in data.files:
                raise KeyError(
                    f"dataset.npz에 {key} 항목이 없습니다."
                )

        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train_raw = np.asarray(data["y_train"]).astype(str)

        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test_raw = np.asarray(data["y_test"]).astype(str)

    # 데이터 형태 확인
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("특징 데이터는 2차원 배열이어야 합니다.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("학습 데이터와 테스트의 특징 수가 다릅니다.")

    if len(X_train) != len(y_train_raw):
        raise ValueError("학습 데이터와 라벨 개수가 다릅니다.")

    if len(X_test) != len(y_test_raw):
        raise ValueError("테스트 데이터와 라벨 개수가 다릅니다.")

    if not np.isfinite(X_train).all():
        raise ValueError("학습 데이터에 NaN 또는 Inf가 있습니다.")

    if not np.isfinite(X_test).all():
        raise ValueError("테스트 데이터에 NaN 또는 Inf가 있습니다.")

    # 문자열 라벨을 정수로 변환
    # XGBoost의 다중 클래스 분류에 사용
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train_raw)

    unknown_labels = (
        set(y_test_raw.tolist())
        - set(encoder.classes_.tolist())
    )

    if unknown_labels:
        raise ValueError(
            f"테스트 데이터에 학습에 없는 라벨이 있습니다: "
            f"{unknown_labels}"
        )

    y_test = encoder.transform(y_test_raw)

    class_names = encoder.classes_.tolist()

    print("=" * 60)
    print("sEMG 3주차 모델 비교 실험")
    print("=" * 60)

    print(f"학습 데이터 크기: {X_train.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    print(f"클래스: {class_names}")

    print("\n[학습 데이터 클래스 분포]")

    labels, counts = np.unique(y_train_raw, return_counts=True)

    for label, count in zip(labels, counts):
        print(f"{label}: {count}개")

    print("\n[테스트 데이터 클래스 분포]")

    labels, counts = np.unique(y_test_raw, return_counts=True)

    for label, count in zip(labels, counts):
        print(f"{label}: {count}개")

    print("\n저장된 학습 및 테스트 데이터를 사용합니다.")
    print("기존 전처리와 데이터 분할은 다시 수행하지 않습니다.")

    return X_train, y_train, X_test, y_test, class_names


# ------------------------------------------------------------
# 3. 모델 구성
# ------------------------------------------------------------

def create_models():

    # 교수님 3주차 실습 자료의 모델 설정을 기준으로 구성
    # 모든 모델은 동일한 학습 데이터와 테스트 데이터 사용

    try:
        from xgboost import XGBClassifier
        import xgboost

    except ImportError as error:
        raise ImportError(
            "XGBoost가 설치되어 있지 않습니다.\n"
            "PowerShell에서 다음 명령어를 실행하세요.\n"
            "python -m pip install xgboost"
        ) from error

    models = {

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
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
        ),
    }

    return models, xgboost.__version__


# ------------------------------------------------------------
# 4. 혼동행렬 저장 및 최대 오분류 표시
# ------------------------------------------------------------

def save_confusion_matrix(
    model_name,
    y_true,
    y_pred,
    class_names,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 셀마다 실제 개수 표시
    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):

            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, col] > threshold
                    else "black"
                ),
                fontsize=9,
            )

    # 대각선을 제외한 최대 오분류 쌍 찾기
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)

    if off_diagonal.max() > 0:

        row, col = np.unravel_index(
            np.argmax(off_diagonal),
            off_diagonal.shape,
        )

        # 최대 오분류 쌍을 빨간 테두리로 표시
        from matplotlib.patches import Rectangle

        rectangle = Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
        )

        ax.add_patch(rectangle)

        error_info = (
            f"{class_names[row]} -> {class_names[col]}: "
            f"{cm[row, col]}개"
        )

    else:
        error_info = "오분류 없음"

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    save_path = OUTPUT_DIR / f"confusion_{filename}.png"

    fig.savefig(
        save_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return cm, error_info


# ------------------------------------------------------------
# 5. 모델 학습 및 테스트 평가
# ------------------------------------------------------------

def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    models,
):

    summary_rows = []
    report_texts = []
    analysis_texts = []

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"[{model_name}] 학습 시작")
        print("=" * 60)

        # 학습 시간 측정
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        train_time = time.perf_counter() - start_time

        # 예측 시간 측정 전 준비
        model.predict(X_test[:min(10, len(X_test))])

        # 전체 테스트 데이터 예측을 5회 반복해 시간 측정
        repeat_count = 5

        start_time = time.perf_counter()

        for _ in range(repeat_count):
            y_pred = model.predict(X_test)

        total_time = time.perf_counter() - start_time

        inference_ms = (
            total_time
            / repeat_count
            / len(X_test)
            * 1000
        )

        # 평가 지표
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        summary_rows.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Train_time_sec": train_time,
            "Inference_ms_per_sample": inference_ms,
        })

        # 클래스별 평가 보고서
        report = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

        report_texts.append(
            f"\n===== {model_name} =====\n{report}\n"
        )

        # 클래스별 평가 결과를 CSV로 저장
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_path = OUTPUT_DIR / (
            "classification_report_"
            + model_name.lower().replace(" ", "_")
            + ".csv"
        )

        with open(
            report_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.writer(file)
            writer.writerow([
                "Class",
                "Precision",
                "Recall",
                "F1-score",
                "Support",
            ])

            for class_name in class_names:

                values = report_dict[class_name]

                writer.writerow([
                    class_name,
                    values["precision"],
                    values["recall"],
                    values["f1-score"],
                    values["support"],
                ])

        # 혼동행렬 저장
        cm, error_info = save_confusion_matrix(
            model_name,
            y_test,
            y_pred,
            class_names,
        )

        # 가장 잘 분류된 클래스 찾기
        class_recalls = {}

        for index, class_name in enumerate(class_names):

            total_actual = cm[index].sum()

            if total_actual > 0:
                class_recalls[class_name] = (
                    cm[index, index] / total_actual
                )

        best_class = max(
            class_recalls,
            key=class_recalls.get,
        )

        # 오분류 분석 결과 저장
        analysis_texts.append(
            f"===== {model_name} =====\n"
            f"가장 높은 재현율을 보인 클래스: {best_class} "
            f"({class_recalls[best_class]:.4f})\n"
            f"최대 오분류 쌍: {error_info}\n"
            f"혼동행렬의 행은 실제 클래스, "
            f"열은 예측 클래스를 의미함.\n"
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print(f"학습 시간: {train_time:.3f}초")
        print(f"추론 시간: {inference_ms:.6f}ms/샘플")
        print(f"최대 오분류 쌍: {error_info}")

    return summary_rows, report_texts, analysis_texts


# ------------------------------------------------------------
# 6. 실험 결과 저장
# ------------------------------------------------------------

def save_results(
    summary_rows,
    report_texts,
    analysis_texts,
    xgboost_version,
    X_train,
    X_test,
    class_names,
):

    # 모델 비교표 CSV
    metrics_path = OUTPUT_DIR / "metrics_holdout.csv"

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # 클래스별 상세 평가 결과
    reports_path = OUTPUT_DIR / "classification_reports.txt"

    reports_path.write_text(
        "\n".join(report_texts),
        encoding="utf-8",
    )

    # 혼동행렬 분석
    analysis_path = OUTPUT_DIR / "confusion_analysis.txt"

    analysis_path.write_text(
        "\n".join(analysis_texts),
        encoding="utf-8",
    )

    # 실행 환경 정보
    info = (
        f"Python version: {sys.version.split()[0]}\n"
        f"NumPy version: {np.__version__}\n"
        f"scikit-learn version: {sklearn.__version__}\n"
        f"XGBoost version: {xgboost_version}\n"
        f"Random state: {RANDOM_STATE}\n"
        f"X_train shape: {X_train.shape}\n"
        f"X_test shape: {X_test.shape}\n"
        f"Classes: {class_names}\n"
        f"Evaluation: hold-out test set\n"
        f"Metrics: Accuracy, Macro Precision, "
        f"Macro Recall, Macro F1\n"
    )

    (OUTPUT_DIR / "run_info.txt").write_text(
        info,
        encoding="utf-8",
    )

    # 최종 비교표 출력
    print("\n" + "=" * 80)
    print("최종 모델 성능 비교")
    print("=" * 80)

    headers = list(summary_rows[0].keys())

    print(" | ".join(headers))

    for row in summary_rows:

        values = []

        for key in headers:

            value = row[key]

            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))

        print(" | ".join(values))

    print("\n결과 저장 완료!")
    print(f"저장 위치: {OUTPUT_DIR}")

    print("\n생성된 파일:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path.name}")


# ------------------------------------------------------------
# 7. 메인 실행
# ------------------------------------------------------------

def main():

    X_train, y_train, X_test, y_test, class_names = (
        load_dataset()
    )

    models, xgboost_version = create_models()

    summary_rows, report_texts, analysis_texts = (
        evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            models,
        )
    )

    save_results(
        summary_rows,
        report_texts,
        analysis_texts,
        xgboost_version,
        X_train,
        X_test,
        class_names,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n실행 중 오류가 발생했습니다.")
        print(f"오류 종류: {type(error).__name__}")
        print(f"오류 내용: {error}")

        raise