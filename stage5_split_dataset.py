
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split

# 경로 및 실험 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "data"
OUT_DIR = BASE_DIR / "week2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000
START = 1000
SEGMENT_LENGTH = 2000
WINDOW_LENGTH = 200
HOP_LENGTH = 100
RANDOM_STATE = 42


def preprocess(csv_path):
    """60 Hz 노치 필터와 20~450 Hz 대역통과 필터를 적용한다."""
    df = pd.read_csv(csv_path)
    signal = df[["Comp Ch 3", "Comp Ch 4"]].to_numpy(dtype=np.float64)

    b_notch, a_notch = iirnotch(60, 30, fs=FS)
    filtered = filtfilt(b_notch, a_notch, signal, axis=0)

    b_band, a_band = butter(
        4, [20, 450], btype="bandpass", fs=FS
    )
    filtered = filtfilt(b_band, a_band, filtered, axis=0)

    return filtered


def cut_segment(signal):
    """각 파일에서 2초 분석 구간을 추출한다."""
    end = START + SEGMENT_LENGTH

    if len(signal) < end:
        raise ValueError(
            f"신호 길이가 부족합니다: 필요한 길이 {end}, 실제 길이 {len(signal)}"
        )

    return signal[START:end]


def make_windows(segment):
    """200 ms 윈도우를 100 ms 간격으로 생성한다."""
    windows = []

    for start in range(
        0, len(segment) - WINDOW_LENGTH + 1, HOP_LENGTH
    ):
        windows.append(segment[start:start + WINDOW_LENGTH])

    return np.asarray(windows, dtype=np.float64)


def td_features(window):
    """두 채널 각각에서 RMS, MAV, WL, ZC, SSC를 계산한다."""
    rms = np.sqrt(np.mean(window ** 2, axis=0))
    mav = np.mean(np.abs(window), axis=0)

    diff = np.diff(window, axis=0)
    wl = np.sum(np.abs(diff), axis=0)

    zc = np.sum(
        ((window[:-1] * window[1:]) < 0)
        & (np.abs(diff) >= 0.01),
        axis=0,
    )

    ssc = np.sum(
        (diff[:-1] * diff[1:]) < 0,
        axis=0,
    )

    return np.concatenate([rms, mav, wl, zc, ssc])


def build_dataset(file_paths, labels):
    """분리된 원본 파일에서 특징과 라벨을 생성한다."""
    features = []
    window_labels = []

    for index, (path, label) in enumerate(
        zip(file_paths, labels), start=1
    ):
        signal = preprocess(path)
        segment = cut_segment(signal)
        windows = make_windows(segment)

        for window in windows:
            features.append(td_features(window))
            window_labels.append(label)

        if index % 25 == 0 or index == len(file_paths):
            print(f"처리 진행: {index}/{len(file_paths)}개 파일")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(window_labels),
    )


def main():
    # 원본 CSV 파일과 클래스 라벨 수집
    files = []
    labels = []

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for csv_path in sorted(class_dir.glob("*.csv")):
            files.append(csv_path)
            labels.append(class_dir.name)

    files = np.asarray(files, dtype=object)
    labels = np.asarray(labels)

    if len(files) == 0:
        raise FileNotFoundError(
            f"CSV 파일을 찾지 못했습니다: {DATA_DIR}"
        )

    unique, counts = np.unique(labels, return_counts=True)

    print("=" * 50)
    print("4. 원본 파일 단위 데이터 분할 확인")
    print("=" * 50)
    print("전체 파일 수:", len(files))
    print("클래스별 파일 수:", dict(zip(unique, counts)))

    # 중요: 윈도우 생성 전에 원본 파일 단위로 분할
    train_files, test_files, train_labels, test_labels = train_test_split(
        files,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_file_set = {str(path.resolve()) for path in train_files}
    test_file_set = {str(path.resolve()) for path in test_files}
    overlap = len(train_file_set & test_file_set)

    print("\n===== 파일 분할 결과 =====")
    print("전체 파일 수:", len(files))
    print("학습 파일 수:", len(train_files))
    print("테스트 파일 수:", len(test_files))
    print("학습 파일과 테스트 파일 중복:", overlap)

    if overlap != 0:
        raise RuntimeError("학습 파일과 테스트 파일이 중복됩니다.")

    print("\n===== 학습 데이터 생성 =====")
    X_train, y_train = build_dataset(train_files, train_labels)

    print("\n===== 테스트 데이터 생성 =====")
    X_test, y_test = build_dataset(test_files, test_labels)

    # 결과 저장
    np.save(OUT_DIR / "stage5_X_train.npy", X_train)
    np.save(OUT_DIR / "stage5_y_train.npy", y_train)
    np.save(OUT_DIR / "stage5_X_test.npy", X_test)
    np.save(OUT_DIR / "stage5_y_test.npy", y_test)

    np.savez(
        OUT_DIR / "stage5_split_info.npz",
        train_files=np.asarray([str(p) for p in train_files]),
        test_files=np.asarray([str(p) for p in test_files]),
    )

    print("\n===== 단계 5 최종 결과 =====")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print(
        "학습 라벨 분포:",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    print(
        "테스트 라벨 분포:",
        dict(zip(*np.unique(y_test, return_counts=True))),
    )
    print("저장 위치:", OUT_DIR.resolve())
    print("확인 완료")


if __name__ == "__main__":
    main()
