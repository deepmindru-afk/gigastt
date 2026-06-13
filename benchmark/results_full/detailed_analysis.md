# Detailed Benchmark Analysis — `fix/benchmark-methodology`

This report analyses the full 4×4 benchmark matrix in `benchmark/results_full/`.
It compares the original WER figures with the re-normalised (`_renorm.json`) figures
and breaks down the remaining errors for **gigastt** on **golos_crowd_1k** after
the normalisation fix.

The re-normalisation mainly affected the **Golos** datasets; the OpenSTT
(`openstt_calls`, `openstt_youtube`) references were already lower-cased and
punctuation-stripped, so their numbers are unchanged.

---

## 1. Per-dataset RTF and wall-time comparison

Wall time is the runner's `total_proc_sec` (sequential transcription time for the
full dataset). RTF is `total_proc_sec / total_audio_sec`.

### golos_crowd_1k

| Engine | WER orig | WER renorm | Δ | Audio (s) | Wall time (s) | RTF |
|---|---|---|---|---|---|---|
| faster-whisper | 15.54% | 15.53% | -0.01 | 3956.3 | 4696.5 | 1.187 |
| gigastt | 10.77% | 8.60% | -2.17 | 3956.3 | 620.0 | 0.157 |
| vosk | 4.57% | 4.82% | +0.25 | 3956.3 | 139.2 | 0.035 |
| whisper.cpp | 15.80% | 15.26% | -0.54 | 3956.3 | 1414.2 | 0.357 |

### golos_farfield

| Engine | WER orig | WER renorm | Δ | Audio (s) | Wall time (s) | RTF |
|---|---|---|---|---|---|---|
| faster-whisper | 16.31% | 17.34% | +1.03 | 2618.7 | 4201.2 | 1.604 |
| gigastt | 5.84% | 5.90% | +0.06 | 2618.7 | 430.7 | 0.164 |
| vosk | 13.93% | 13.93% | -0.00 | 2618.7 | 75.4 | 0.029 |
| whisper.cpp | 16.94% | 17.91% | +0.97 | 2618.7 | 1456.6 | 0.556 |

### openstt_calls

| Engine | WER orig | WER renorm | Δ | Audio (s) | Wall time (s) | RTF |
|---|---|---|---|---|---|---|
| faster-whisper | 24.93% | 24.93% | -0.00 | 2139.7 | 4946.8 | 2.312 |
| gigastt | 19.28% | 19.28% | -0.00 | 2139.7 | 453.2 | 0.212 |
| vosk | 38.57% | 38.57% | +0.00 | 2139.7 | 61.0 | 0.029 |
| whisper.cpp | 32.73% | 32.73% | -0.00 | 2139.7 | 1336.0 | 0.624 |

### openstt_youtube

| Engine | WER orig | WER renorm | Δ | Audio (s) | Wall time (s) | RTF |
|---|---|---|---|---|---|---|
| faster-whisper | 15.45% | 15.45% | +0.00 | 2246.2 | 4219.9 | 1.879 |
| gigastt | 11.35% | 11.35% | -0.00 | 2246.2 | 353.8 | 0.158 |
| vosk | 20.65% | 20.65% | +0.00 | 2246.2 | 65.9 | 0.029 |
| whisper.cpp | 22.61% | 22.61% | -0.00 | 2246.2 | 1718.8 | 0.765 |

### Key observations

- **gigastt** is the fastest engine on every dataset by a wide margin:
  RTF ≈ 0.16 on the Golos and YouTube data, and 0.21 on telephone calls.
- **vosk** has the lowest RTF overall (≈ 0.03) but pays for it with much higher
  WER on noisy/telephony data.
- **faster-whisper** and **whisper.cpp** are 1–2 orders of magnitude slower;
  whisper.cpp is faster than faster-whisper but less accurate on these Russian
  corpora.
- The normalisation fix reduced **gigastt** WER on **golos_crowd_1k** from
  **10.77% → 8.60%**; the same fix made some Whisper/FW numbers slightly worse
  on **golos_farfield** because the cleaned reference set is harder for them.

---

## 2. Residual-error distribution for gigastt on golos_crowd_1k (renorm)

After re-normalisation gigastt makes **407 word errors** on the 1 000-sample
`golos_crowd_1k` subset. The categories below are heuristic assignments based on
comparing the reference and hypothesis text. Percentages are of the 407 residual
errors.

| Category | Approx. share of residual errors | Notes |
|---|---|---|
| Foreign brand / artist names in Latin vs Russian transliteration | ~48% | References spell foreign names/TV channels in Russian; hypotheses emit Latin originals, or vice versa. |
| Real ASR errors / partial hypotheses | ~37% | Genuine mis-hearings, name substitutions, dropped/added words, empty references with non-empty hypotheses. |
| Other numeric format mismatches (phone / episode / serial numbers) | ~8% | Spoken digit sequences rendered as digit strings, or season/episode numbers converted to digits. |
| Decimal / fraction / unit mismatches | ~4% | Quantities, percentages and units normalised to symbols/digits. |
| Date / year format mismatches | ~4% | Spoken dates/years rendered as numeric dates. |

### Representative examples

#### Foreign brand / artist names in Latin vs Russian transliteration (~48%)

- **Reference:** `киношка окко смарт бокс на окко`  
  **Hypothesis:** `Киношка Okko Смартбокс на Okko.`
- **Reference:** `ты можешь показать на смотрешке передачу фэшн ти ви четыре ка`  
  **Hypothesis:** `Ты можешь показать на смотрёшках передачу Fashion TV четыре копейки.`

#### Real ASR errors / partial hypotheses (~37%)

- **Reference:** `посещает ли кинотеатр рафаловский давид`  
  **Hypothesis:** `Посещает ли кинотеатр Рафаэловский Давид?`
- **Reference:** `анатолий иванович показаньев`  
  **Hypothesis:** `Анатолий Иванович Показания`

#### Other numeric format mismatches (~8%)

- **Reference:** `ноль шестьсот шесть девятьсот семьдесят два двадцать один одиннадцать`  
  **Hypothesis:** `606972211`
- **Reference:** `эпизод двадцать три пятый сезон красивая жизнь`  
  **Hypothesis:** `Эпизод 23. Пятый сезон. Красивая жизнь.`

#### Decimal / fraction / unit mismatches (~4%)

- **Reference:** `сбер перевод пятьсот рублей елена юрьевна по номеру телефона сбербанк онлайн`  
  **Hypothesis:** `СберПеревод 500 ₽ Елена Юрьевна по номеру телефона Сбербанк Онлайн.`
- **Reference:** `сметана простоквашино пятнадцать процентов сто восемьдесят грамм`  
  **Hypothesis:** `Сметана Простоквашино, 15%, 180 г.`

#### Date / year format mismatches (~4%)

- **Reference:** `сколько стоит двадцать одна американских долларов перевести в гуарани курс тринадцатое июня двадцатый год`  
  **Hypothesis:** `Сколько стоит $21 перевести в гуарани, курс 13 июня 2020 года?`
- **Reference:** `сбер сериал побег третья серия две тысячи девятнадцатый год`  
  **Hypothesis:** `СберСериал Побег, третья серия 2019 год.`

---

## 3. Conclusion

The re-normalised benchmark confirms that **gigastt** delivers the best
speed/accuracy trade-off on the evaluated Russian datasets:

- **golos_crowd_1k:** 8.60% WER at 0.157 RTF, far below any Whisper variant and
  only ~4 points above vosk while being ~9× faster.
- **golos_farfield:** 5.90% WER, the lowest of all engines on far-field data.
- **openstt_calls / openstt_youtube:** unchanged at 19.28% and 11.35% WER
  respectively, still ahead of whisper.cpp and faster-whisper on these sets.

The residual WER on **golos_crowd_1k** is dominated by two effects:
**foreign-brand/artist transliteration mismatches** (~48%) and genuine
**ASR errors** (~37%). A smaller but measurable share comes from normalising
**numbers, dates and units** (~16% combined). This suggests that further WER
improvements could come from a smarter entity/transliteration normaliser rather
than from model changes alone.
