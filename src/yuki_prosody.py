import io
import wave
import numpy as np


def _estimate_pitch(frame, sample_rate):
    frame = frame.astype(np.float32)
    frame -= np.mean(frame)

    rms = float(np.sqrt(np.mean(frame * frame)))
    if rms < 0.0025:
        return None

    frame *= np.hanning(len(frame))

    # FFT autocorrelation: faster and more stable for voiced speech.
    n = len(frame)
    nfft = 1 << (2 * n - 1).bit_length()

    spectrum = np.fft.rfft(frame, n=nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n]

    if corr[0] <= 1e-12:
        return None

    corr /= corr[0]

    min_hz = 70.0
    max_hz = 400.0

    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(n - 1, int(sample_rate / min_hz))

    if max_lag <= min_lag:
        return None

    region = corr[min_lag:max_lag + 1]

    # Prefer an actual local periodicity peak.
    peaks = []
    for i in range(1, len(region) - 1):
        if region[i] > region[i - 1] and region[i] >= region[i + 1]:
            peaks.append(i)

    if peaks:
        best = max(peaks, key=lambda i: region[i])
    else:
        best = int(np.argmax(region))

    lag = min_lag + best
    confidence = float(corr[lag])

    if confidence < 0.12:
        return None

    pitch = sample_rate / lag

    if not (min_hz <= pitch <= max_hz):
        return None

    return pitch


def analyze_wav(wav_data):
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError("Prosody analyzer currently expects 16-bit PCM WAV.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if len(audio) == 0:
        return {
            "duration": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "pitch_hz": None,
            "pitch_variation": None,
        }

    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))

    frame_size = int(sample_rate * 0.080)
    hop_size = int(sample_rate * 0.020)

    pitches = []

    for start in range(0, max(0, len(audio) - frame_size + 1), hop_size):
        frame = audio[start:start + frame_size]
        pitch = _estimate_pitch(frame, sample_rate)

        if pitch is not None:
            pitches.append(pitch)

    if pitches:
        pitches = np.asarray(pitches, dtype=np.float32)

        # Robust center.
        center = float(np.median(pitches))

        # Fold common octave errors toward the median.
        corrected = []
        for value in pitches:
            value = float(value)

            while value > center * 1.75:
                value /= 2.0

            while value < center / 1.75:
                value *= 2.0

            corrected.append(value)

        corrected = np.asarray(corrected, dtype=np.float32)

        # Re-center after octave correction.
        pitch_hz = float(np.median(corrected))

        # Ignore extreme tracking frames when measuring variation.
        lo = np.percentile(corrected, 10)
        hi = np.percentile(corrected, 90)

        stable = corrected[
            (corrected >= lo) &
            (corrected <= hi)
        ]

        if len(stable) >= 3:
            pitch_variation = float(np.std(stable))
        else:
            pitch_variation = 0.0
    else:
        pitch_hz = None
        pitch_variation = None

    return {
        "duration": round(duration, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "pitch_hz": None if pitch_hz is None else round(pitch_hz, 1),
        "pitch_variation": (
            None if pitch_variation is None
            else round(pitch_variation, 1)
        ),
    }

class ProsodyTracker:
    def __init__(self, alpha=0.15, calibration_turns=3):
        self.alpha = alpha
        self.calibration_turns = calibration_turns
        self.calibration_count = 0
        self.baseline_rms = None
        self.baseline_pitch = None
        self.baseline_variation = None

    def _align_pitch(self, pitch):
        """Choose the octave-equivalent F0 closest to the session baseline."""
        if self.baseline_pitch is None:
            return pitch

        reference = self.baseline_pitch

        candidates = [
            pitch / 4.0,
            pitch / 2.0,
            pitch,
            pitch * 2.0,
            pitch * 4.0,
        ]

        candidates = [
            value for value in candidates
            if 70.0 <= value <= 400.0
        ]

        if not candidates:
            return pitch

        return min(
            candidates,
            key=lambda value: abs(__import__("math").log2(value / reference)),
        )

    def update(self, metrics, has_transcript=True):
        # Ignore recordings that are probably silence/noise.
        if (
            not has_transcript
            or metrics["rms"] < 0.008
            or metrics["pitch_hz"] is None
            or metrics["pitch_variation"] is None
        ):
            return {
                "valid": False,
                "state": "unknown",
                "arousal": 0.0,
            }

        rms = metrics["rms"]
        pitch = metrics["pitch_hz"]
        variation = metrics["pitch_variation"]

        # Correct likely octave/harmonic jumps relative to this session.
        pitch = self._align_pitch(pitch)

        # Build the personal baseline from the first few valid turns.
        if self.calibration_count < self.calibration_turns:
            if self.calibration_count == 0:
                self.baseline_rms = rms
                self.baseline_pitch = pitch
                self.baseline_variation = max(variation, 1.0)
            else:
                n = self.calibration_count

                self.baseline_rms = (
                    self.baseline_rms * n + rms
                ) / (n + 1)

                self.baseline_pitch = (
                    self.baseline_pitch * n + pitch
                ) / (n + 1)

                self.baseline_variation = (
                    self.baseline_variation * n + max(variation, 1.0)
                ) / (n + 1)

            self.calibration_count += 1

            return {
                "valid": True,
                "state": "neutral",
                "arousal": 0.0,
                "calibrating": True,
            }

        rms_ratio = rms / max(self.baseline_rms, 1e-6)
        pitch_ratio = pitch / max(self.baseline_pitch, 1e-6)
        variation_ratio = variation / max(self.baseline_variation, 1.0)

        # Positive = more animated/energetic than personal baseline.
        # Score rising and falling prosody separately.
        # This makes the detector less sensitive to microphone-distance
        # changes in RMS.

        rms_up = max(0.0, rms_ratio - 1.0)
        pitch_up = max(0.0, pitch_ratio - 1.0)
        variation_up = max(0.0, variation_ratio - 1.0)

        rms_down = max(0.0, 1.0 - rms_ratio)
        pitch_down = max(0.0, 1.0 - pitch_ratio)
        variation_down = max(0.0, 1.0 - variation_ratio)

        lively_score = (
            0.55 * rms_up
            + 0.30 * pitch_up
            + 0.10 * variation_up
        )

        calm_score = (
            0.10 * rms_down
            + 0.55 * pitch_down
            + 0.35 * variation_down
        )

        arousal = lively_score - calm_score

        # Keep downstream prosody controls bounded.
        arousal = max(-1.0, min(1.0, arousal))

        if arousal > 0.70:
            state = "animated"
        elif arousal > 0.15:
            state = "lively"
        elif arousal < -0.30:
            state = "calm"
        elif arousal < -0.08:
            state = "relaxed"
        else:
            state = "neutral"

        # Slowly adapt baseline so ordinary changes do not permanently
        # redefine the user's normal voice.
        a = self.alpha
        self.baseline_rms = (1 - a) * self.baseline_rms + a * rms
        self.baseline_pitch = (1 - a) * self.baseline_pitch + a * pitch
        self.baseline_variation = (
            (1 - a) * self.baseline_variation + a * max(variation, 1.0)
        )

        return {
            "valid": True,
            "state": state,
            "arousal": round(float(arousal), 3),
        }
