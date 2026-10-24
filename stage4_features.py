
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "week2_results"
WINDOW_FILE = RESULT_DIR / "stage3_windows.npy"
FEATURE_FILE = RESULT_DIR / "stage4_features.npy"


def td_features(window):
    """한 윈도우의 두 채널에서 시간 영역 특징 10개를 계산한다."""
    features = []

    for channel in range(window.shape[1]):
        signal = window[:, channel]

        # RMS: 제곱 평균의 제곱근
        rms = np.sqrt(np.mean(signal ** 2))

        # MAV: 평균 절댓값
        mav = np.mean(np.abs(signal))

        # WL: 파형 길이
        wl = np.sum(np.abs(np.diff(signal)))

        # ZC: 영점 교차 횟수
        zc = np.sum(
            (signal[:-1] * signal[1:] < 0)
        )

        # SSC: 기울기 부호 변화 횟수
        diff_signal = np.diff(signal)
        ssc = np.sum(
            (diff_signal[:-1] * diff_signal[1:] < 0)
        )

        features.extend([rms, mav, wl, zc, ssc])

    return np.asarray(features, dtype=np.float64)


def main():
    print("=" * 45)
    print("3. 시간 영역 특징 추출 확인")
    print("=" * 45)

    if not WINDOW_FILE.exists():
        print(f"오류: 윈도우 파일을 찾을 수 없습니다: {WINDOW_FILE}")
        return

    windows = np.load(WINDOW_FILE)

    # 모든 윈도우에 특징 추출 함수를 적용한다.
    features = np.asarray(
        [td_features(window) for window in windows]
    )

    np.save(FEATURE_FILE, features)

    print(f"입력 윈도우 배열 모양: {windows.shape}")
    print(f"특징 데이터 모양: {features.shape}")
    print(f"윈도우당 특징 개수: {features.shape[1]}")
    print("특징 순서: 채널 1의 RMS, MAV, WL, ZC, SSC,")
    print("           채널 2의 RMS, MAV, WL, ZC, SSC")
    print("\n첫 번째 특징 벡터:")
    print(features[0])
    print(f"\nNaN 포함 여부: {np.isnan(features).any()}")
    print(f"Inf 포함 여부: {np.isinf(features).any()}")
    print(f"\n저장 파일: {FEATURE_FILE}")
    print("확인 완료")


if __name__ == "__main__":
    main()
