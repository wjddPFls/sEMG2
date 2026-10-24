
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt

# 실험 설정
FS = 1000
NOTCH_FREQ = 60
NOTCH_Q = 30
LOWCUT = 20
HIGHCUT = 450

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "data" / "A" / "a (1).csv"
OUTPUT_DIR = BASE_DIR / "week2_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(x, fs=FS):
    """60Hz 노치 필터와 20~450Hz 대역통과 필터 적용."""
    # 1. 60Hz 전원선 잡음 제거
    bn, an = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=fs)
    x_notch = filtfilt(bn, an, x, axis=0)

    # 2. 20~450Hz 대역통과 필터
    b, a = butter(
        4,
        [LOWCUT / (fs / 2), HIGHCUT / (fs / 2)],
        btype="band"
    )
    x_filtered = filtfilt(b, a, x_notch, axis=0)

    return x_filtered


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    # 원본 CSV 읽기
    df = pd.read_csv(CSV_PATH)
    expected_columns = ["Comp Ch 3", "Comp Ch 4"]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"예상한 채널이 없습니다. 현재 열 이름: {df.columns.tolist()}"
        )

    x = df[expected_columns].to_numpy(dtype=np.float64)

    if x.shape != (3000, 2):
        raise ValueError(f"예상하지 못한 데이터 모양: {x.shape}")

    if not np.isfinite(x).all():
        raise ValueError("원본 데이터에 NaN 또는 무한값이 있습니다.")

    # 필터 적용
    xf = preprocess(x)

    # 표준편차 확인
    print("사용 파일:", CSV_PATH.name)
    print("필터 전 모양:", x.shape)
    print("필터 후 모양:", xf.shape)
    print("필터 전 표준편차:", x.std(axis=0).round(6))
    print("필터 후 표준편차:", xf.std(axis=0).round(6))

    # 필터 전후 파형 비교
    time = np.arange(len(x)) / FS

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    for ch in range(2):
        axes[ch].plot(time, x[:, ch], label="Before filter", alpha=0.7)
        axes[ch].plot(time, xf[:, ch], label="After filter", alpha=0.8)
        axes[ch].set_title(expected_columns[ch])
        axes[ch].set_xlabel("Time (seconds)")
        axes[ch].set_ylabel("Amplitude")
        axes[ch].legend()
        axes[ch].grid(True)

    plt.tight_layout()
    figure_path = OUTPUT_DIR / "stage1_filter.png"
    plt.savefig(figure_path, dpi=200)
    plt.close(fig)

    # 필터 전후 스펙트럼 비교
    freqs = np.fft.rfftfreq(len(x), d=1 / FS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    for ch in range(2):
        before = np.abs(np.fft.rfft(x[:, ch])) / len(x)
        after = np.abs(np.fft.rfft(xf[:, ch])) / len(x)

        axes[ch].plot(freqs, before, label="Before filter")
        axes[ch].plot(freqs, after, label="After filter")
        axes[ch].set_xlim(0, FS / 2)
        axes[ch].set_title(expected_columns[ch])
        axes[ch].set_xlabel("Frequency (Hz)")
        axes[ch].set_ylabel("Amplitude")
        axes[ch].legend()
        axes[ch].grid(True)

    plt.tight_layout()
    spectrum_path = OUTPUT_DIR / "stage1_spectrum.png"
    plt.savefig(spectrum_path, dpi=200)
    plt.close(fig)

    # 필터 결과 저장
    np.save(OUTPUT_DIR / "stage1_filtered.npy", xf)

    print("\n저장 완료:")
    print(figure_path)
    print(spectrum_path)
    print(OUTPUT_DIR / "stage1_filtered.npy")


if __name__ == "__main__":
    main()
