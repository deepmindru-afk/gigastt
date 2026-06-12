# WER Re-normalization Summary

| Dataset | Engine | Old WER | Old CI | New WER | New CI | Δ WER |
|---|---|---|---|---|---|---|
| golos_crowd_1k | faster-whisper | 15.54 | 14.06–16.96 | 15.53 | 13.94–17.10 | -0.01 |
|  | gigastt | 10.77 | 9.17–12.16 | 8.60 | 7.51–9.66 | -2.17 |
|  | vosk | 4.57 | 3.82–5.33 | 4.82 | 4.03–5.60 | +0.25 |
|  | whisper.cpp | 15.80 | 14.34–17.26 | 15.26 | 13.74–16.71 | -0.54 |
| golos_farfield | faster-whisper | 16.31 | 14.71–17.89 | 17.34 | 15.62–19.07 | +1.03 |
|  | gigastt | 5.84 | 5.05–6.71 | 5.90 | 5.09–6.83 | +0.06 |
|  | whisper.cpp | 16.94 | 15.40–18.51 | 17.91 | 16.29–19.57 | +0.97 |
