# WER Re-normalization Summary

| Dataset | Engine | Old WER | Old CI | New WER | New CI | Δ WER |
|---|---|---|---|---|---|---|
| golos_crowd_1k | faster-whisper | 15.54 | 14.06–16.96 | 14.75 | 13.33–16.17 | -0.79 |
|  | gigastt | 10.77 | 9.17–12.16 | 8.77 | 7.67–9.85 | -2.00 |
|  | vosk | 4.57 | 3.82–5.33 | 4.82 | 4.03–5.60 | +0.25 |
|  | whisper.cpp | 15.80 | 14.34–17.26 | 14.98 | 13.61–16.41 | -0.82 |
| golos_farfield | faster-whisper | 16.31 | 14.71–17.89 | 17.32 | 15.61–19.05 | +1.01 |
|  | gigastt | 5.84 | 5.05–6.71 | 5.94 | 5.14–6.86 | +0.10 |
|  | whisper.cpp | 16.94 | 15.40–18.51 | 17.93 | 16.31–19.58 | +0.99 |
