
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "week2_results" / "stage1_filtered.npy"
OUTPUT_DIR = BASE_DIR / "week2_results"

FS = 1000
START = 1000  # 1초 지점
LEN = 2000    # 2초 구간


def cut(x, start=START, n=LEN):
    if start < 0 or start + n > len(x):
        raise ValueError(
            f"구간 범위를 벗어났습니다: "
            f"시작={start}, 길이={n}, 전체 길이={len(x)}"
        )
    return x[start:start + n]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    xf = np.load(INPUT_PATH)

    print("입력 데이터 모양:", xf.shape)

    seg = cut(xf)

    print("세그먼트 모양:", seg.shape)
    print("시작 시간:", START / FS, "초")
    print("종료 시간:", (START + LEN) / FS, "초")

    np.save(OUTPUT_DIR / "stage2_segment.npy", seg)

    time = np.arange(len(seg)) / FS + START / FS

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    for ch in range(seg.shape[1]):
        axes[ch].plot(time, seg[:, ch])
        axes[ch].set_title(f"EMG Channel {ch + 1}")
        axes[ch].set_xlabel("Time (seconds)")
        axes[ch].set_ylabel("Amplitude")
        axes[ch].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage2_segment.png", dpi=200)
    plt.close(fig)

    print("저장 완료:", OUTPUT_DIR / "stage2_segment.npy")
    print("저장 완료:", OUTPUT_DIR / "stage2_segment.png")


if __name__ == "__main__":
    main()
