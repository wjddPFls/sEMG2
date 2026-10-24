
from pathlib import Path

import numpy as np

# 현재 파일 위치 기준으로 최종 데이터셋을 불러온다.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "week2_results" / "dataset.npz"


def main():
    print("=" * 50)
    print("5. 최종 데이터셋 요약")
    print("=" * 50)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}"
        )

    with np.load(DATASET_PATH) as data:
        required_keys = ["X_train", "y_train", "X_test", "y_test"]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            raise KeyError(f"데이터셋에 필요한 항목이 없습니다: {missing_keys}")

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        print("파일명:", DATASET_PATH.name)
        print("\n[데이터 배열 모양]")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

        print("\n[학습 데이터 클래스별 분포]")
        train_classes, train_counts = np.unique(
            y_train, return_counts=True
        )
        for label, count in zip(train_classes, train_counts):
            print(f"클래스 {label}: {count}개")

        print("\n[테스트 데이터 클래스별 분포]")
        test_classes, test_counts = np.unique(
            y_test, return_counts=True
        )
        for label, count in zip(test_classes, test_counts):
            print(f"클래스 {label}: {count}개")

        print("\n[표준화 정보]")
        for key in ["scaler_mean", "scaler_scale"]:
            if key in data:
                print(f"{key}: {data[key].shape}")
            else:
                print(f"{key}: 파일에 없음")

    print("\n최종 데이터셋 확인 완료")
    print("저장 파일:", DATASET_PATH)


if __name__ == "__main__":
    main()
