# HIBIKI.CPP

Local-first, low-latency voice AI built around LiquidAI LFM2.5-Audio and llama.cpp.

## Status

Early developer preview.

- Native Windows Vulkan server
- AMD GPU acceleration
- Hands-free microphone input
- Streaming voice output
- Persistent multi-turn context
- Silence-based turn detection
- TTFT / TTFA / audio latency benchmarks

## Client

```powershell
py -m pip install -r requirements.txt
py .\src\hibiki_client.py --base-url http://127.0.0.1:8080/v1 --mode interleaved --hands-free
```

## Models

Model weights are not included in this repository.

Current target: LiquidAI LFM2.5-Audio-1.5B.

## Project goal

Turn the current working setup into a portable Windows voice application that does not require Python, Visual Studio, WSL, or the Vulkan SDK at runtime.

## License

Third-party license notices will be included before the first public release.

## AI Assistance Disclosure

Portions of this codebase were generated with assistance from OpenAI's GPT-5.6 Sol.
All generated code included in this repository was subsequently reviewed, tested, and accepted by a human maintainer before publication.
