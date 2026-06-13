# Top-10 residual WER errors for gigastt on golos_crowd_1k after renormalization

Overall WER: 8.60% (CI 7.51%–9.66%), total errors: 407, total ref words: 4732

Whitespace-only references are excluded from this view; they are filtered by `load_manifest` in new runs.
These samples show why the WER stays above the hoped-for 3–5% range.
The dominant remaining source of errors is not numbers but foreign names / brands / artist names that gigastt outputs in original Latin spelling while the reference uses Russian transliteration.

## 1. errors=6 ref_words=7 audio=~/.gigastt/benchmarks/golos_wav/6e09f5d0b88999611da712050945549d.wav

- **Reference:** включи гуд лайф джи изи и кехлани
- **Hypothesis:** Включи Good Life G Eazy I Kehlani.
- **Normalized ref:** включи гуд лайф джи изи и кехлани
- **Normalized hyp:** включи good life g eazy i kehlani

## 2. errors=5 ref_words=11 audio=~/.gigastt/benchmarks/golos_wav/a660f244bc2e55685984398dd57fa7f8.wav

- **Reference:** ты можешь показать на смотрешке передачу фэшн ти ви четыре ка
- **Hypothesis:** Ты можешь показать на смотрёшках передачу Fashion TV четыре копейки.
- **Normalized ref:** ты можешь показать на смотрешке передачу фэшн ти ви 4 ка
- **Normalized hyp:** ты можешь показать на смотрешках передачу fashion тв 4 копейки

## 3. errors=5 ref_words=1 audio=~/.gigastt/benchmarks/golos_wav/38bf4f3c89e0507ee921a669fec12e9d.wav

- **Reference:** ноль триста восемь триста восемнадцать триста четыре шестьдесят девять
- **Hypothesis:** Ноль, 308, 318, 304, 69
- **Normalized ref:** 30831830469
- **Normalized hyp:** 0 308 318 304 69

## 4. errors=4 ref_words=6 audio=~/.gigastt/benchmarks/golos_wav/1fde5bb67dfa702d2685ec3a71fc6290.wav

- **Reference:** киношка окко смарт бокс на окко
- **Hypothesis:** Киношка Okko Смартбокс на Okko.
- **Normalized ref:** киношка окко смарт бокс на окко
- **Normalized hyp:** киношка okko смартбокс на okko

## 5. errors=4 ref_words=9 audio=~/.gigastt/benchmarks/golos_wav/a790301a15650756451d359ecf48a4a0.wav

- **Reference:** покажи на смотрешке телеканал фэшн ти ви четыре ка
- **Hypothesis:** Покажи на смотрешке телеканал Fashion TV четыре копейки.
- **Normalized ref:** покажи на смотрешке телеканал фэшн ти ви 4 ка
- **Normalized hyp:** покажи на смотрешке телеканал fashion тв 4 копейки

## 6. errors=4 ref_words=12 audio=~/.gigastt/benchmarks/golos_wav/9d901ea90dc0e685b88cea25a30f0895.wav

- **Reference:** сколько стоит двадцать одна американских долларов перевести в гуарани курс тринадцатое июня двадцатый год
- **Hypothesis:** Сколько стоит $21 перевести в гуарани, курс 13 июня 2020 года?
- **Normalized ref:** сколько стоит 21 американских перевести в гуарани курс 13 июня 20 год
- **Normalized hyp:** сколько стоит $ 21 перевести в гуарани курс 13 июня 2020 года

## 7. errors=4 ref_words=4 audio=~/.gigastt/benchmarks/golos_wav/2e5b765fe39f77a0c0ff458085e3fc83.wav

- **Reference:** тичинг менс фэшэн ютюб
- **Hypothesis:** Teaching Mells Fashion YouTube.
- **Normalized ref:** тичинг менс фэшэн ютюб
- **Normalized hyp:** teaching mells fashion ютуб

## 8. errors=4 ref_words=5 audio=~/.gigastt/benchmarks/golos_wav/dbe488f494ee8a1f2a316c2ff520b1f9.wav

- **Reference:** шоу эмбер вулф гэйм ютьюб
- **Hypothesis:** Шоу AmberWolf Game YouTube.
- **Normalized ref:** шоу эмбер вулф гэйм ютьюб
- **Normalized hyp:** шоу amberwolf game ютуб

## 9. errors=4 ref_words=4 audio=~/.gigastt/benchmarks/golos_wav/f6e193c03f6d742326edd17a2da053e0.wav

- **Reference:** афина включи лаки люк
- **Hypothesis:** 
- **Normalized ref:** афина включи лаки люк
- **Normalized hyp:** 

## 10. errors=4 ref_words=7 audio=~/.gigastt/benchmarks/golos_wav/5cd8cf78e1f435acdc008c41648584a6.wav

- **Reference:** закажи пожалуйста пепси два с половиной литра
- **Hypothesis:** Закажи, пожалуйста, Pepsi 2,5 л.
- **Normalized ref:** закажи пожалуйста пепси 2 с половиной литра
- **Normalized hyp:** закажи пожалуйста pepsi 2 5 л
