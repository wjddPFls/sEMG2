
import numpy as np
from pathlib import Path

# 현재 파이썬 파일이 있는 폴더를 기준으로 결과 파일을 찾는다.
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "week2_results"
WINDOW_FILE = RESULT_DIR / "stage3_windows.npy"


def main():
    print("=" * 45)
    print("2. 윈도우 배열 확인")
    print("=" * 45)

    if not WINDOW_FILE.exists():
        print(f"오류: 윈도우 파일을 찾을 수 없습니다.\n경로: {WINDOW_FILE}")
        return

    windows = np.load(WINDOW_FILE)

    print(f"불러온 파일: {WINDOW_FILE.name}")
    print(f"전체 윈도우 배열 모양: {windows.shape}")
    print(f"윈도우 개수: {windows.shape[0]}")
    print(f"윈도우 길이: {windows.shape[1]} samples")
    print(f"윈도우 길이: {windows.shape[1] / 1000 * 1000:.0f} ms")
    print(f"채널 수: {windows.shape[2]}")
    print(f"첫 번째 윈도우 모양: {windows[0].shape}")
    print(f"NaN 포함 여부: {np.isnan(windows).any()}")
    print(f"Inf 포함 여부: {np.isinf(windows).any()}")

    print("\n첫 번째 윈도우의 앞부분:")
    print(windows[0, :5, :])

    print("\n확인 완료")


if __name__ == "__main__":
    main()
