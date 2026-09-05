import time
import json
from pathlib import Path

from openai import OpenAI
import yuki_client as h
import yuki_prosody as prosody


AUDIO_URL = "http://127.0.0.1:8083/v1"
BRAIN_URL = "http://127.0.0.1:8084/v1"

# Conversation tuning
# off   = do not repeat what the user said
# brief = short spoken confirmation
# full  = repeat the recognized utterance before replying
CONFIG_PATH = Path(r"C:\Yuki\config\yuki.json")

try:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

ECHO_MODE = str(CONFIG.get("echo_mode", "off")).lower()

if ECHO_MODE not in {"off", "brief", "full"}:
    ECHO_MODE = "off"


BRAIN_SYSTEM = (
    "\u81ea\u7136\u306a\u65e5\u672c\u8a9e\u3067\u7c21\u6f54\u306b\u4f1a\u8a71\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
    "\u30e6\u30fc\u30b6\u30fc\u306e\u767a\u8a00\u3092\u305d\u306e\u307e\u307e\u7e70\u308a\u8fd4\u3055\u305a\u3001"
    "\u81ea\u7136\u306a\u76f8\u3065\u3061\u3084\u4e00\u8a00\u306e\u611f\u60f3\u3067\u76f4\u63a5\u8fd4\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
    "\u81ea\u5206\u306b\u8eab\u4f53\u3084\u5929\u6c17\u306e\u4f53\u611f\u304c\u3042\u308b\u3088\u3046\u306a\u8868\u73fe\u306f\u907f\u3051\u3066\u304f\u3060\u3055\u3044\u3002"
    "\u539f\u52471\u6587\u3001\u5fc5\u8981\u306a\u5834\u5408\u306e\u307f2\u6587\u3002"
    "\u5b9a\u578b\u7684\u306a\u6848\u5185\u3084\u7de0\u3081\u306e\u6587\u53e5\u306f\u4f7f\u308f\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"
    "\n"
    "\u5fc5\u305a\u65e5\u672c\u8a9e\u3060\u3051\u3067\u8fd4\u7b54\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
    "\u82f1\u8a9e\u306e\u6587\u3084\u82f1\u8a9e\u306e\u30d5\u30ec\u30fc\u30ba\u306b\u5207\u308a\u66ff\u3048\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"

)



def build_brain_system(tone_state):
    if tone_state == "lively":
        return (
            BRAIN_SYSTEM
            + "\n"
            + "\u30e6\u30fc\u30b6\u30fc\u306e\u8a71\u3057\u65b9\u306f"
              "\u5c11\u3057\u6d3b\u767a\u3067\u3059\u3002"
              "\u8fd4\u7b54\u3082\u5c11\u3057\u3060\u3051\u660e\u308b\u304f"
              "\u81ea\u7136\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
              "\u5927\u3052\u3055\u306a\u8868\u73fe\u3084"
              "\u4e0d\u5fc5\u8981\u306a\u611f\u5606\u7b26\u306f\u907f\u3051\u3066\u304f\u3060\u3055\u3044\u3002"
              "\u3053\u306e\u5224\u5b9a\u81ea\u4f53\u306b\u306f"
              "\u8a00\u53ca\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"
        )

    if tone_state == "animated":
        return (
            BRAIN_SYSTEM
            + "\n"
            + "\u30e6\u30fc\u30b6\u30fc\u306e\u8a71\u3057\u65b9\u306f"
              "\u6d3b\u767a\u3067\u30a8\u30cd\u30eb\u30ae\u30c3\u30b7\u30e5\u3067\u3059\u3002"
              "\u8fd4\u7b54\u3082\u5c11\u3057\u660e\u308b\u304f\u6d3b\u767a\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
              "\u5927\u3052\u3055\u306b\u306f\u305b\u305a\u3001"
              "\u3053\u306e\u5224\u5b9a\u81ea\u4f53\u306b\u306f\u8a00\u53ca\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"
        )

    if tone_state == "relaxed":
        return (
            BRAIN_SYSTEM
            + "\n"
            + "\u30e6\u30fc\u30b6\u30fc\u306e\u8a71\u3057\u65b9\u306f"
              "\u5c11\u3057\u843d\u3061\u7740\u3044\u3066\u3044\u307e\u3059\u3002"
              "\u8fd4\u7b54\u3082\u5c11\u3057\u7a4f\u3084\u304b\u3067"
              "\u81ea\u7136\u306a\u8abf\u5b50\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
              "\u5927\u3052\u3055\u306b\u306f\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"
        )

    if tone_state == "calm":
        return (
            BRAIN_SYSTEM
            + "\n"
            + "\u30e6\u30fc\u30b6\u30fc\u306e\u8a71\u3057\u65b9\u306f"
              "\u843d\u3061\u7740\u3044\u3066\u3044\u307e\u3059\u3002"
              "\u8fd4\u7b54\u3082\u7a4f\u3084\u304b\u3067\u63a7\u3048\u3081\u306a\u8abf\u5b50\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
              "\u3053\u306e\u5224\u5b9a\u81ea\u4f53\u306b\u306f\u8a00\u53ca\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002"
        )

    return BRAIN_SYSTEM



def get_tts_sampler(tone_state):
    if tone_state == "relaxed":
        return 0.76, 56

    if tone_state == "calm":
        return 0.70, 48

    if tone_state == "lively":
        return 0.86, 70

    if tone_state == "animated":
        return 0.95, 80

    return 0.80, 64


def build_spoken_reply(transcript, reply):
    mode = ECHO_MODE.strip().lower()

    if mode == "brief":
        return "\u3046\u3093\u3001\u805e\u3053\u3048\u305f\u3088\u3002" + reply

    if mode == "full":
        return transcript + " " + reply

    return reply


def main():
    audio = OpenAI(
        base_url=AUDIO_URL,
        api_key="dummy",
    )

    brain = OpenAI(
        base_url=BRAIN_URL,
        api_key="dummy",
    )

    recorder = h.AudioRecorder()
    tracker = prosody.ProsodyTracker()

    if not recorder.available:
        raise SystemExit("No microphone available.")

    print()
    print("YUKI.CPP LIVE BRIDGE")
    print("8083  Japanese Audio  ASR + TTS")
    print("8084  LFM2.5-8B-A1B  Brain")
    print()
    print("Mic -> ASR -> 8B -> TTS -> Speakers")
    print("Press Ctrl+C to stop.")
    print()

    while True:
        try:
            samples = recorder.record_until_silence()

            if not samples:
                continue

            wav_data = recorder.to_wav_bytes()

            if not wav_data:
                continue

            metrics = prosody.analyze_wav(wav_data)

            pitch = (
                "-"
                if metrics["pitch_hz"] is None
                else f'{metrics["pitch_hz"]:.1f}Hz'
            )

            variation = (
                "-"
                if metrics["pitch_variation"] is None
                else f'{metrics["pitch_variation"]:.1f}Hz'
            )

            print(
                f'[prosody '
                f'duration={metrics["duration"]:.2f}s '
                f'rms={metrics["rms"]:.5f} '
                f'peak={metrics["peak"]:.5f} '
                f'pitch={pitch} '
                f'variation={variation}]'
            )

            # --------------------------------------------------
            # 1. Speech -> Japanese text
            # --------------------------------------------------
            print("\n=== ASR ===")

            stream = h.create_stream_single_shot(
                audio,
                "asr",
                wav_data=wav_data,
                max_tokens=256,
            )

            transcript, _ = h.process_stream(stream)
            transcript = transcript.strip()

            if not transcript:
                print("[No transcript]")
                continue

            # --------------------------------------------------
            # 2. Japanese text -> 8B brain
            # --------------------------------------------------
            tone = tracker.update(
                metrics,
                has_transcript=True,
            )

            print(
                f'[tone '
                f'state={tone["state"]} '
                f'arousal={tone["arousal"]:+.3f}]'
            )

            print("\n=== BRAIN ===")

            brain_t0 = time.perf_counter()

            response = brain.chat.completions.create(
                model="",
                messages=[
                    {
                        "role": "system",
                        "content": build_brain_system(tone["state"]),
                    },
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ],
                max_tokens=768,
            )

            choice = response.choices[0]
            reply = (choice.message.content or "").strip()
            brain_time = time.perf_counter() - brain_t0

            print(reply)
            print(
                f"[brain {brain_time:.3f}s | "
                f"finish {choice.finish_reason}]"
            )

            if not reply:
                print("[Brain returned no spoken content]")
                continue

            # --------------------------------------------------
            # 3. 8B reply -> Japanese speech
            # --------------------------------------------------
            spoken_reply = build_spoken_reply(transcript, reply)

            print("\n=== TTS ===")

            player = h.AudioPlayer()
            player.start()

            try:
                audio_temperature, audio_top_k = get_tts_sampler(
                    tone["state"]
                )

                print(
                    f"[tts-prosody "
                    f"state={tone['state']} "
                    f"temp={audio_temperature:.2f} "
                    f"top_k={audio_top_k}]"
                )

                stream = h.create_stream_single_shot(
                    audio,
                    "tts",
                    text=spoken_reply,
                    max_tokens=1024,
                    audio_temperature=audio_temperature,
                    audio_top_k=audio_top_k,
                )

                h.process_stream(stream, player)

            finally:
                player.stop()

            print()

        except KeyboardInterrupt:
            print("\nYUKI stopped.")
            break

        except Exception as exc:
            print(f"\n[Bridge error: {exc}]")


if __name__ == "__main__":
    main()





