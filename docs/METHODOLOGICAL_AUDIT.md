# ExoBiome — audyt metodologiczny i inwentaryzacja repozytorium
---

## 0. Werdykt w skrócie

**Obecny stan wyników jest niepublikowalny — nie z powodu uproszczeń hackathonowych, a z powodu trzech
faktów empirycznych, z których każdy osobno kończy recenzję. Czwarty problem, opisany pod tabelą, jest
konstrukcyjny, nie wynikowy.**

| # | Fakt | Dowód |
|---|---|---|
| 1 | Wszystkie 685 widm POSEIDON to **linie stałe** (685/685 wierszy `transit_depth_noiseless` ma dokładnie jedną unikalną wartość na 218 binów). Oś główna projektu mierzy reakcję modeli na wejście bez informacji. | `audit/a01` |
| 2 | Wszystkie modele cross-generator mają **ujemną skill** względem predyktora stałego (2.894): quantum −0.111, noquant −0.133, winner −0.193.| `audit/a02` |
| 3 | Porównanie „ExoBiome 0.30 bije SOTA 0.55" wynika z **różnych wejść**. Po zrównaniu konwencji przewaga spada z 1,85× do **1,29×** na ramieniu `median`, które zespół faktycznie opublikował (`holdout_metrics.json: point_estimate = "median"`), a w konwencji zaszumionej **odwraca się** na **1,74×** na korzyść flow. Na ramieniu `mean` odpowiednio 1,35× i 1,76×. | `audit/a03` |


**Część kwantowa jest martwa nie z powodu wyniku, a z powodu konstrukcji:** gałąź „kwantowa" ma 69 434
parametry, z czego **24 są kwantowe** (0,0093 % modelu), jej wyjście to druga głowa klasyczna nad tym samym
kontekstem, do której obwód dokłada 8 z 264 wymiarów wejścia, przemnożoną przez `|tanh(gate)|·scale ∈ [0.0178, 0.0292]`.
Taka architektura nie może wyprodukować **ani** pozytywnej, **ani** rzetelnie negatywnej odpowiedzi na
postawione pytanie badawcze (`audit/a06`).

---

## 0a. Czego dotyczą zarzuty: modelu czy aparatury pomiarowej

Większość ustaleń
tego audytu dotyczy danych, pomiaru i sposobu porównania — nie jakości samego modelu. Zarzut wobec aparatury nie jest zarzutem wobec
architektury, a mieszanie ich prowadziłoby do wniosku „model jest zły", którego dane nie popierają.

### Co model faktycznie robi na osi, która nie jest zepsuta

Zbiór ADC2023 jest **realny i nieuszkodzony**: dane organizatorów challenge'u, nasz split, 4143
wiersze holdoutu, a rekonstrukcja pipeline'u przechodzi z marginesem 11×. Drabina
baseline'ów na tej osi (`a02`, `a26`, przebieg `20260729`):

| klasa | co dokładnie | mRMSE [dex] | skill vs stała treningowa |
|---|---|---:|---:|
| podłoga | stała treningowa | 1.4404 | 0 |
| **proste baseline'y** | tylko tabela aux, 8 kolumn | 1.4407 | **−0.0002** |
| **proste baseline'y** | tylko widmo, ridge | 1.0174 | +0.294 |
| **proste baseline'y** | tylko widmo, GBM | 0.4409 | +0.694 |
| baseline organizatorów | CNN z `ADC2023-baseline` | 0.6500 | +0.549 |
| model porównawczy | NSF z rodziny zwycięzcy, `median`, wejście +N(0,σ) — konwencja publikowana | 0.5523 | +0.617 |
| model porównawczy | ten sam NSF, **wejście zrównane** z ExoBiome (widmo jak w pliku) | 0.3846 | **+0.733** |
| **model projektu** | **ExoBiome (skala 1.0)** | **0.2994** | **+0.792** |

Proste baseline'y to dwa nietuningowane modele ogólnego przeznaczenia, po to, by pokazać, ile z widma
da się wyciągnąć bez sieci: **ridge** to regresja liniowa z karą za duże współczynniki (osiem linii
`numpy`, ta sama receptura co w zespołowym `baseline_smoke.py`), a **GBM** to zespół płytkich drzew
decyzyjnych, który łapie nieliniowości, ale traktuje biny widma jako niezależne kolumny. Dwa różne
learnery zamiast jednego po to, żeby wynik nie zależał od wyboru estymatora.

Wniosek, który wolno postawić: **jako regresor punktowy pięciu abundancji na widmach ADC model działa
i jest najlepszym wynikiem zmierzonym w tym repozytorium** — 1,47× lepszym od GBM na tym samym widmie,
przy tabeli aux.

Dwa wiersze modelu porównawczego są podane celowo. Wobec
**publikowanej** konwencji NSF (widmo + N(0,σ)) różnica skillu wynosi +0,175, ale to porównanie dwóch
różnych zadań i właśnie ono jest przedmiotem K3 (więcej tam). Wobec **zrównanego** wejścia różnica spada do **+0,06
skillu** (0.2994 vs 0.3846, iloraz 1,29×). Tylko uwaga: to są dwa wywołania tego samego wytrenowanego modelu, nie dwa treningi `a03` ładuje raz `best_model_by_mrmse.pt` i różni tylko `sample_noise` na wejściu (`preprocessing.py:146-149`).
Zastrzeżenie, którego nie wolno pominąć: `train.py` woła trening z `sample_noise=True` **zawsze**, bez
flagi konfiguracyjnej (l. 161, 181, 219, 253, 267) — model nigdy nie widział podczas nauki czystego
widma bez dolosowanego szumu. Wiersz „wejście +N(0,σ)" ocenia go więc w rozkładzie treningowym; wiersz
„wejście zrównane" ocenia go **poza** tym rozkładem. „Zrównane wejście" oznacza tylko, że oba modele
dostają to samo widmo — nie oznacza, że oba są oceniane w warunkach, do których zostały wytrenowane.

### Dlaczego awaria na crossgenie nie jest zarzutem wobec modelu

Na zbiorze POSEIDON skill wynosi: ExoBiome **−0,111**, noquant **−0,133**, a **ridge −0,019 i GBM
−0,127** dopasowane niezależnie. Cztery modele z trzech różnych klas zawodzą identycznie.

**Granica tego argumentu, nazwana wprost.** Wszystkie cztery ramiona są **uczone na `tau/train`
i oceniane na POSEIDON**, więc zerowy skill jest zgodny
z **dwiema** hipotezami: dane nie noszą informacji **lub/oraz** żaden model nie przenosi się między
generatorami. Rozdziela je dopiero pomiar **na samych danych, bez udziału modelu**: 685/685 wierszy
`transit_depth_noiseless` ma jedną wartość na 218 binów (`a01`, kryterium bezprogowe).
Widmo stałe po λ nie może nieść informacji o składzie. Kolejnym punktem byłaby ewaluacja na dobrze wygenerowanych danych,
aby wykluczyć drugą część hipotezy.

Odwrotnie na `tau_val`, gdzie dane mają strukturę: oba warianty ExoBiome biją GBM (1.605, skill +0,444),
choć skromniej niż na ADC — wariant **bez kwantów** 1.423 (+0,507) i wariant kwantowy 1.449 (+0,498),
czyli 1,13× i 1,11×. Lepszy z tej pary jest wariantem klasycznym.

### Podział ustaleń

| dotyczy aparatury pomiarowej — nie mówi nic o jakości modelu | dotyczy samego modelu |
|---|---|
| **K1** widma POSEIDON stałe po λ | **K5** faza wspólna 2 z ~24 epok; wysłany checkpoint sprzed odmrożenia backbone'u |
| **K11** generator TauREx bez CIA | **K6** 24 z 69 434 parametrów gałęzi są kwantowe (0,0093 %) |
| **K3** benchmark porównuje dwie konwencje wejścia | **K7** bramka zero-init, niezbieżna, CO₂ o przeciwnym znaku |
| **K4** metryki raportowane w punkcie pracy, w którym model nie był walidowany | **K9** 5 z 7 parametrów benchmarku |
| **K2** brak trywialnego baseline|||
| **K10** metryka projektu ≠ metryka benchmarku| |

### Co zostało zrobione dobrze

1. **Dwuetapowy trening z klasycznym punktem startowym — i faza kwantowa, która zrobiła to, co miała.**
   `config.json` flagowego runu wskazuje `init_checkpoint_path → stage1_classical/best_model.pt`, a gałąź
   kwantowa dochodzi w etapie 2 z zamrożonym backbone'em (`quantum_backbone_freeze_epochs = 6`). To jest
   właściwy szkielet ablacji. Co więcej, ta faza
   **wykonała się w całości i działała**: przez wszystkie sześć zamrożonych epok walidacja poprawiała się
   monotonicznie (0,29333 → 0,29081, `history.csv`). Zarzut „gałąź kwantowa nie zdążyła się wytrenować"
   jest **wykluczony**. Wada K6/K7 nie leży ani w projekcie eksperymentu, ani w przebiegu tej fazy, tylko
   w tym, że wkład jest o dwa rzędy wielkości za mały, by cokolwiek zmierzyć. Osobno (problem z K5) faza **wspólna** (backbone odmrożony) dostała 2 epoki z ~24 zaplanowanych i destabilizowała się
   natychmiast (+13,0 % na epoce 7, dokładnie w momencie `backbone_frozen 1 → 0`), a wysłany checkpoint
   jest **sprzed niej**.
2. **Osobny wariant bez kwantów** (`models/taurex_exobiome_without_quant`)
3. **Reimplementacja modelu odniesienia z rodziny zwycięzcy ADC2023**, wytrenowana lokalnie na naszym
   splicie, z zapisanym checkpointem i metrykami, nie zacytowana z pracy.
4. **Determinizm modelu.** Dwa wywołania na tym samym checkpoincie dają `max |p₁ − p₂| = 0.000e+00`.
   Jedyne wcześniej niepowtarzalne wielkości, czyli estymator punktowy NSF, losowany z posterioru flow
   (`a24:185`), są juz zaseedowane po stronie audytu i potwierdzone bit-identyczne między
   przebiegami.

### Na czym model był trenowany — precyzyjnie, bo to jest źródłem połowy sporu

- **Pewne, z kodu:** trening ExoBiome **nie zawiera żadnej augmentacji szumem**. Skan całej ścieżki
  `models/ariel_exobiome/*.py` po `randn`, `np.random`, `torch.normal`, `augment`, `add_noise` daje
  jedno trafienie — `np.random.seed(seed)` w `training.py:127`. `config.json` flagowego runu nie ma
  żadnego parametru szumu. Model konsumuje widmo tak, jak leży w pliku, a wektor σ dostaje jako
  52 osobne kanały wejścia (`constants.py`, `MODEL_SPECTRAL_CHANNELS`).
- **Udokumentowane w źródle pierwotnym:** organizatorzy i zwycięzcy challenge'u opisują udostępnione
  widma ADC jako **idealne**. Szum nie jest w nie wliczony, tablica σ jest dostarczana osobno
  (arXiv:2309.09337, §1.2, *Noise*).


**Dla wniosku K3 nie ma to znaczenia** i to jest istotne: cokolwiek jest zapisane w pliku, ExoBiome
konsumował to bez zmian, a baseline NSF dokładał pełną dodatkową σ w treningu i w ewaluacji
(`preprocessing.py:146-149`, `train.py:161,181,219,253,267`). Asymetria konwencji jest pewna z kodu po
obu stronach, niezależnie od tego, ile szumu jest w samych widmach.

---

## 1. Jak audytowano i jak to odtworzyć

Kod audytowy leży w `audit/`. Każdy check ładuje wyłącznie zacommitowane artefakty, nie modyfikuje kodu
modeli ani treningu, i zapisuje samoopisujący się rekord JSON (commit gita, wersje środowiska, sha256 wejść,
werdykt) do `reports/audit/<data>/`.

```bash
cd ../hack4sages
./.venv-qml/bin/python audit/run_all.py
```

```bash
./.venv-qml/bin/python audit/run_all.py --fast
```

`run_all.py` kończy się kodem ≠ 0, jeśli którykolwiek check ma status FAIL. Pełna specyfikacja checków,
jest w `audit/README.md`.

**Walidacja kodu audytu.** Liczby w K3, K4 i K9 nie są odczytane z artefaktów z repo, musiały być przeliczone
rekonstrukcją pipeline'u ExoBiome w `audit_lib.py`, która od nowa czyta surowy HDF5, stosuje zapisane
skalery i przepuszcza dane przez model. Gdyby ta rekonstrukcja różniła się od oryginału, wszystkie trzy
ustalenia byłyby bezwartościowe, dlatego jest sprawdzana osobnym checkiem, zanim cokolwiek się na niej
oprze. `a27_pipeline_fidelity` porównuje ją z zacommitowanymi predykcjami **wiersz po wierszu**, na
wszystkich 4143 wierszach × 5 gazach:

| wielkość | wartość |
|---|---:|
| kryterium PASS (tolerancja) | 1,0e-4 |
| największa różnica na 20 715 komórkach | **8,9e-6** |
| p99 różnic | 1,5e-6 |
| wierszy powyżej tolerancji | **0 z 4143** |

Najgorsza komórka leży więc **jedenaście razy poniżej progu**, przy którym check by upadł — a gdyby
rekonstrukcja była błędna, rozbieżności byłyby rzędu samych mierzonych efektów (0,003–0,007), nie 10⁻⁶.


### 1. Reprodukowalność harnessu

Pełny przebieg `run_all.py` (`20260729`): **21 checków**, jeden katalog
(`reports/audit/20260729/`), `is_partial_run: False`, **0 ERROR**, wynik
`FAIL 17 / WARN 1 / INFO 1 / PASS 2`.

**Harness w pełni odtwarzalny.** Jedyna wcześniej niepowtarzalna wielkość (estymator punktowy NSF)
brała się z niezaseedowanego losowania posterioru (`IndependentNSF.sample`, `model.py:54`, bez
generatora). Naprawione w audycie: `a03`/`a24` seedują globalny RNG (`POSTERIOR_SAMPLE_SEED = 42`,
zapisane w payloadzie) przed każdym losowaniem, bez ingerencji w kod zespołu. Potwierdzone:
dwa niezależne uruchomienia `a03` dają **54 wielkości, 0 różnic**. Reszta harnessu była deterministyczna
od początku (`max|p₁−p₂| = 0` dla ExoBiome, zliczenia strukturalne niezależne od losowania).

> Seed daje **odtwarzalność**, nie dokładność: estymator to nadal jedno losowanie ze 128 próbek
> posterioru. Rozrzut po trzech seedach szumu: `mrmse_std` = 0,0009 (`mean`) / 0,0006 (`median`).

---

## 2. Tabela inwentaryzacyjna

Status na przebiegu `reports/audit/20260729/` (21 checkow, `is_partial_run: False`, **0 ERROR**).

| id | problem | dowód | wynik checku |
|---|---|---|---|
| K1 | Widma POSEIDON są stałe po długości fali | `a01_spectral_variation` | FAIL |
| K1(b) | Smoke baseline zespołu **był uruchomiony**, ustalenie postawił zarchiwizowany audyt | `a29_smoke_baseline_recovery` | PASS (ustalenie zawężone — patrz K1(b)) |
| K1(c) | Diagnoza: skrypt do generacji atmosfery poprawny. |  `d01` stage 1 | zawężone, decyzja 2026-07-27 |
| K2 | Brak baseline'u trywialnego; ujemna skill na osi głównej |  `a02_trivial_baseline` | FAIL |
| K3 | Niezgodne konwencje wejścia w benchmarku vs SOTA + niedotrenowany model zwycięzców |  `a03_input_convention`, `a05`, `a09` | FAIL |
| K4 | Raportowane metryki przy `quantum_scale=1.0`, selekcja przy 0.5 | `a04_quantum_scale_provenance` | FAIL |
| K5 | Flagowy checkpoint z epoki 6; faza wspólna przerwana ręcznie w 8/30 epok | `a05_training_completeness` | FAIL |
| K6 | Wkład kwantowy nieodseparowalny od 69 410 parametrów klasycznych | `a06_param_accounting` | FAIL |
| K7 | Model sam wyciszył kwant: bramka jest zero-init, a jej wartość końcowa niezerowa |  `a07_gate_dynamics` | FAIL |
| K9 | Model przewiduje 5 z 7 parametrów benchmarku; T sprzężona z abundancjami 4,14× silniej niż gazy między sobą | `a15_target_completeness` | FAIL |
| K10(c) | Oficjalny baseline ADC2023 jest w repo (`models/adc_baseline/`), używa MC-dropout i produkuje rozkład|  lektura + `ucl-exoplanets/ADC2023-baseline` | PROVEN |
| K11 | TauREx generuje bez CIA: brak głównego kontinuum w H₂/He | analiza kodu, m. in. `data/crossgen_biosignatures/taurex_backend.py:154-157` | — |
| K10 | Raportowana metryka (mRMSE) **nie jest** metryką ADC2023 (0,8·KS na 7 marginalnych + 0,2·widmowa); model punktowy jest na niej niepunktowalny | literatura + `a24_official_metrics` | FAIL |

Dodatkowe problemy:
| id | problem | dowód | wynik checku |
|---|---|---|---|
| P1 | `taurex_fmpe`: holdout ≡ validation | `a10_split_integrity` | FAIL |
| P2 | Brak zbioru testowego in-domain na TauREx |  `a10_split_integrity` | FAIL |
| P4 | Suite 7 testów: kierunek prawdopodobny, liczby zepsute | czytanie kodu | — |
| P7 | Jeden seed w całym projekcie; brak korekty na wielokrotność |  - | FAIL |
| P9 | Generatory różnią się opacity/rozdzielczością/CIA/He | publikacje | — |

---

## 3. Problemy krytyczne — dowody

### K1. Widma POSEIDON są stałe po długości fali

**Gdzie znaleziony.** `data/TauREx set/spectra.h5`, pole `transit_depth_noiseless`, wszystkie 685
wierszy POSEIDON. Modele czytają `transit_depth_noisy` (`models/taurex_exobiome/dataset.py:287`).
Zdania, które to ustalenie podważa, stoją w `reports/taurex_model_comparison.md` (cały ranking
transferu). Kod generatora:
`poseidon_backend.py:78-88` — `from POSEIDON.core import compute_spectrum, make_atmosphere, ...`, czyli
import prawdziwej biblioteki, nie reimplementacja fizyki — oraz `poseidon_backend.py:51`, gdzie
powstaje siatka ciśnień `np.geomspace(PRESSURE_MIN_BAR, PRESSURE_MAX_BAR, PRESSURE_LEVELS)`.
Hipoteza wtórna dotyczy `data/scripts/bootstrap_poseidon_input_data.sh:10`.

**Wyjaśnienie.**

```
poseidon  transit_depth_noiseless  relvar_med=0.000e+00  const rows=685/685  feat/sigma=0.000  frac SNR>1=0.000
poseidon  transit_depth_noisy      relvar_med=2.144e-03  const rows=  0/685  feat/sigma=0.998  frac SNR>1=0.488
tau       transit_depth_noiseless  relvar_med=1.289e-02  const rows=0/41423 feat/sigma=6.417  frac SNR>1=0.922
tau       transit_depth_noisy      relvar_med=1.377e-02  const rows=0/41423 feat/sigma=6.488  frac SNR>1=0.997
```

`transit_depth_noiseless` dla POSEIDON: **jedna skalarna głębokość tranzytu powtórzona 218 razy** (wiersz 0:
`0.01698546` × 218; `n_unique_values_median = 1`, `bit_constant_rows = 685/685`). Widmo, które nie zależy
od długości fali, nie może nieść informacji o składzie atmosfery, niezależnie 
przyczyny.

Pole, które modele czytają, ma `apparent_amplitude_over_sigma_median = 0.998`,
tylko 48,8 % wierszy ma SNR > 1, a SNR > 3 **żaden** — czyli **cała zmienność tych „widm" to dodany szum**.
Dla porównania tau, to samo pole: `6.488`, SNR > 1 w 99,7 % wierszy, SNR > 3 w 72,1 %.


**Mechanizm maskujący w preprocessingu.**  Bramka, która tego nie łapie: `validate_dataset.py:78-128, chociaz to nie jedyne miejsce, które tego nie złapało. Po prostu **pipeline modelu nie mógł tego zgłosić, bo usuwa dokładnie tę
informację w pierwszym kroku**. ExoBiome dzieli każde widmo przez jego własną średnią
(`models/ariel_exobiome/dataset.py:414-416`, ten sam kod w `models/taurex_exobiome/dataset.py`);
reimplementacja zwycięzcy odejmuje średnią, dzieli przez odchylenie z `clamp(min=1e-6)` i podaje średnią
oraz odchylenie jako dwie osobne cechy (`models/adc_winner_on_ariel/preprocessing.py:151-155`). W obu
wariantach człon stały wypada z wektora widmowego, więc widmo płaskie **po normalizacji jest
nierozróżnialne skalą od zdrowego**: zostaje 218 kanałów (dla ADC 52) szumu o poprawnej amplitudzie.

**Zakres tego ustalenia.** Generator jest cienką nakładką na prawdziwą bibliotekę (`from POSEIDON.core
import …`), a nie reimplementacją fizyki, więc defekt musi leżeć w sposobie wywołania API albo w torze
danych opacity. Które dokładnie, to juz K1(c) i **nie wchodzi do tego ustalenia**: K1 stwierdza
fakt (widma nie zależą od λ) i jego konsekwencję dla wyników, nie przyczynę.

Jedno wykluczenie należy do tego ustalenia, bo dotyczy samego pomiaru: **nie jest to błąd kształtu
zwracanej tablicy.** Obie ścieżki rebinningu by go złapały: `rebin_spectrum` waliduje `values.shape`
jawnie, a `fixed_native_matrix.dot(...)` rzuciłby wyjątek na niezgodności wymiarów.

**Skąd próg `1e-4` i czy jest wystarczający.** Dla tego werdyktu **nie ma znaczenia**: decyduje kryterium
bezprogowe („zero wierszy bit-stałych"), a POSEIDON ma 685/685 wierszy bit-stałych przy
`rel_variation = 0.000e+00` **dokładnie**. Żadna wartość progu tego nie odwraca, próg rozstrzyga tylko
przypadki graniczne, a tu nie ma ani jednego. 


Jednak od tego próg da się wyprowadzić fizycznie, a nie tylko przyjąć: relatywna amplituda cechy transmisyjnej na jedną skalę wysokości wynosi `2H/R_p`,
gdzie `H = kT/(μg)` (dodaj ta pracke o widmach). Policzone z etykiet tych 685 planet (`temperature_k`, `log_g_cgs`,
`planet_radius_rjup`, μ z `vmr_h2`/`vmr_he`): μ mediana **2,314**, `H` mediana **208 km** (zakres 38–974),
a `2H/R_p` mediana **5,42e-3** przy **minimum 9,48e-4** na całym zbiorze. Próg leży więc **9,5× poniżej
najsłabszego fizycznie możliwego sygnału** w tym zbiorze, cokolwiek prawdziwego musi go przekroczyć.
Zmierzone tau ma `p01 = 1,99e-3`, czyli 20× nad progiem; POSEIDON **0,0**.

Ograniczenie samego checku: `a01` nie wykrywa *struktury o złej
amplitudzie*. Generator z brakującym członem opacity (ustalenie K11) przechodzi ten check z definicji.

Literatura. arXiv:1409.2312 (Waldmann i in. 2014, TauREx I, Taurex III[dodaj cytat]) jest referencją dla konfiguracji
forward-modelu (CIA, ciśnienie referencyjne, listy linii) czyli dla tego, jak powinna wyglądać
zgodność dwóch generatorów; bez niej „gap" mierzy różnicę konfiguracji, nie fizyki. arXiv:2210.06564
(Ward i in. 2022, RNPE) daje ramę teoretyczną dla cross-generator gap jako misspecyfikacji, ale wolno
jej użyć **dopiero po** naprawie K1: misspecyfikacja wymaga dwóch poprawnych modeli, a nie jednego
poprawnego i jednego zwracającego stałą.

Dlaczego dyskwalifikuje: unieważnia każdą liczbę cross-generator w repo (3.2156, 3.2796, 3.4531, 2.8946,
wszystkie „gapy", cały `reports/taurex_model_comparison.md`). Awaria procesowa: `validate_dataset.py:78-128` sprawdza kształt, skończoność,
dodatniość, zakresy priorów i prewalencję — nie sprawdza, czy widmo zależy od długości fali.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a01_spectral_variation.py` |
| komenda | `./.venv-qml/bin/python audit/a01_spectral_variation.py` |
| kryterium PASS | zero wierszy bit-stałych ∧ mediana feature/σ > 1 ∧ mediana `std_bins/|mean_bins|` > 1e-4; sprawdzany jest **każdy** wiersz |
| wartość oczekiwana | Status `FAIL`, `failing_generators = [poseidon]`. POSEIDON `transit_depth_noiseless`: `bit_constant_rows = 685/685`, `rel_variation_median = 0.0`, `feature_amplitude_over_sigma_median = 0.0`, `first_row_head = 0.016985462978482246`; `transit_depth_noisy`: `apparent_amplitude_over_sigma_median = 0.998`, `frac_rows_snr_gt_1 = 0.488`, `frac_rows_snr_gt_3 = 0.0`. tau, wszystkie 41 423 wiersze, to samo pole: `6.488` / `0.997` / `0.721` |
| co by to obaliło | choćby dwie różne wartości w wierszu `transit_depth_noiseless` przy medianie feature/σ > 1 — czyli istotna zmienność po λ przekraczająca szum. Wtedy oś cross-generator byłaby ważna i wszystkie liczby POSEIDON wracają. Ustalenia **nie** obala 218 unikalnych wartości w `transit_depth_noisy` (to szum, `feat/sigma = 0.998`) ani ustalenie przyczyny (K1(c)) |
| rekord | `reports/audit/20260729/a01_spectral_variation.json` |

### K1(b). Pierwsze sygnały były w tym repozytorium przed audytem

**Gdzie znaleziony.** `data/TauREx set/baseline_smoke.json` wraz z `data/TauREx set/baseline_poseidon_predictions.csv`. **Nieśledzone w gicie w chwili przekazania; do indeksu wprowadził je ten audyt** (`4c431db`, 2026-07-27 22:09, po dodaniu negacji w `.gitignore:46-50` w `61adef1`).

**Wyjaśnienie.**

Co ten plik mówi — cztery niezależne sygnały awarii, wszystkie w JSON-ie.

Sygnał 1 — mRMSE wobec predyktora stałego. Pierwszy sygnał, ze cos moze być nie tak.

| zbiór | pole w JSON | mRMSE | baseline stały | skill |
|---|---|---:|---:|---:|
| tau/val | `val_rmse` | 2,6147 | 2,8852 | **+0,0938** |
| poseidon/test | `test_rmse` | 2,8954 | 2,8940 | **−0,0005** |


Sygnał 2 — odchylenie predykcji (z `baseline_poseidon_predictions.csv`).

| gaz | sd(pred) | sd(etykiet) | iloraz |
|---|---:|---:|---:|
| log10_vmr_h2o | 0,0684 | 2,8742 | 0,024 |
| log10_vmr_co2 | 0,0736 | 2,9670 | 0,025 |
| log10_vmr_co | 0,0461 | 2,8448 | 0,016 |
| log10_vmr_ch4 | 0,0751 | 2,9426 | 0,026 |
| log10_vmr_nh3 | 0,0578 | 2,8436 | 0,020 |

Predykcje mają **2,2 % rozrzutu etykiet** i skupiają się przy **−7,014**, czyli przy średniej prioru
`U(−12, −2)`. Nie są literalnie stałe — mają 684 unikalne wartości — ale mieszczą się w pasmie
**[−7,53; −6,77]** o szerokości 0,26 σ etykiet. Predyktor zapadł się więc do wąskiego pasma wokół
średniej prioru, ignorując wejście.

Sygnał 3 — dokładność binarna równa prewalencji klas, co do czwartego miejsca po przecinku.

| gaz | `test_binary_accuracy` | prewalencja z `manifest.json` | różnica | `val_binary_accuracy` | prewalencja tau | różnica |
|---|---:|---:|---:|---:|---:|---:|
| H₂O | 0,5766 | 0,5766 | **+0,0000** | 0,6357 | 0,6034 | +0,032 |
| CO₂ | 0,5956 | 0,5956 | **+0,0000** | 0,6463 | 0,6037 | +0,043 |
| CO | 0,6234 | 0,6234 | **+0,0000** | 0,5987 | 0,5988 | −0,000 |
| CH₄ | 0,6088 | 0,6088 | **+0,0000** | 0,6564 | 0,6006 | +0,056 |
| NH₃ | 0,6058 | 0,6058 | **+0,0000** | 0,6393 | 0,5962 | +0,043 |

Mechanizm wynika z sygnału 2: całe pasmo predykcji **[−7,53; −6,77]** leży **powyżej** progu obecności
`−8,0`, więc **każdy** z 685 wierszy dostaje dla każdego gazu etykietę „obecny". Dokładność klasyfikatora,
który zawsze orzeka ze jest, jest równa udziałowi wierszy, w których gaz faktycznie jest czyli
prewalencji. Zgodność do czwartego miejsca jednocześnie na pięciu gazach jest tego deterministyczną
konsekwencją. Sygnał 3 jest więc sprawdzeniem sygnału 2, a nie osobnym dowodem: pokazuje
ten sam kolaps w wielkości, która nie wymaga wyboru żadnego baseline'u.

Sygnał 4 — metadane potwierdzają, że to ten bundel: `feature_dim = 219` (218 binów + `sigma_ppm`),
`train_rows = 37281`, `val_rows = 4142`, `test_rows = 685`.

**Wniosek** Oba artefakty są
zadeklarowane w **kodzie produkcyjnym** jako pola `DatasetPaths`
(`data/crossgen_biosignatures/constants.py:88-89`, śledzone od 2026-03-11), a `a29` znajduje **osiem**
konsumentów odczytujących je po ścieżce, wszystkie śledzone w gicie i starsze od tego audytu. Do tego
zarchiwizowana próba audytu zespołu (`archive/…/11_baseline_comparison.md`) opisuje ten artefakt jako
mean-predictor i oznacza go jako CRITICAL. Mimo tego ani `reports/taurex_model_comparison.md`, ani plan
prac nie zawierają śladu tej diagnozy: ranking transferu został zbudowany i ogłoszony bez niej.


**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a29_smoke_baseline_recovery.py` weryfikacja przez lekturę `data/TauREx set/baseline_smoke.json`, `data/TauREx set/baseline_poseidon_predictions.csv` i `data/TauREx set/manifest.json` |
| komenda | `./.venv-qml/bin/python audit/a29_smoke_baseline_recovery.py` |
| kryterium PASS | Kryterium jest odwrócone, bo ten check ma **szansę obalić** ustalenie, a nie je potwierdzić. **PASS = ustalenie obalone**, jeśli zachodzi **którykolwiek** z czterech warunków; **FAIL = ustalenie stoi**, gdy nie zachodzi żaden. Warunki poniżej |
| ↳ (a) | brakuje któregoś artefaktu w `data/TauREx set/` → nie ma czego czytać, więc nie ma ustalenia |
| ↳ (b) | CSV nie odtwarza `test_rmse` z JSON-a, per gaz, w granicy `1e-9` → artefakty nie pochodzą z jednego przebiegu na tym bundlu |
| ↳ (c) | brak kontrastu znaków: `sign(skill[tau_val]) == sign(skill[poseidon_test])` → dostarczone liczby same z siebie nie wskazywały POSEIDON-a |
| ↳ (d) | dowolny plik poza `data/TauREx set/`, `audit/`, `docs/` i `reports/audit/` odczytuje artefakty po nazwie → wynik **został skonsumowany**, więc teza „nikt tego nie przeczytał" nie może wejść do ustalenia w tym brzmieniu |
| wartość oczekiwana | `test_rmse` = `2.8738 / 2.9661 / 2.8482 / 2.9477 / 2.8410` (średnia 2,8954) wobec baseline'u stałego 2,8940 → skill **−0,0005**; `val_rmse` średnio 2,6147 wobec 2,8852 → skill **+0,0938**; `test_binary_accuracy` = prewalencja z `manifest.json` co do czwartego miejsca dla wszystkich pięciu gazów (0,5766 / 0,5956 / 0,6234 / 0,6088 / 0,6058); `feature_dim = 219`, `train_rows = 37281`, `val_rows = 4142`, `test_rows = 685`; konsumentów spoza bundla/`audit/`/`docs/` — **8**, wszystkie śledzone w gicie, wszystkie starsze od audytu |
| co by to obaliło | każdy z (a)–(d) z osobna — dowolny obala ustalenie w jego pierwotnym brzmieniu. W tym przebiegu zaszło **wyłącznie (d)**, więc `a29` = `PASS`. (a)–(c) nie zaszły: artefakty są kompletne, spójne między sobą i wykazują kontrast znaku skill między `taurex_val` a `poseidon_test` — ta część ustalenia stoi niepodważona. Obalony jest tylko fragment o nieodczytaniu wyniku, dlatego ustalenie figuruje wyżej w brzmieniu „nie znalazło odbicia w wynikach", a nie „nikt tego nie przeczytał" |
| rekord | `reports/audit/20260729/a29_smoke_baseline_recovery.json` |

---

### K1(c). Diagnoza Etapu 1: atmosfera jest zdrowa, opacity jest **nasycone**

**Gdzie znaleziony.** `poseidon_backend.py:51` (siatka ciśnień) i `poseidon_backend.py:78-88`
(wywołania `make_atmosphere` / `compute_spectrum`), zdiagnozowane przez
`audit/d01_poseidon_diagnosis.py --stage 1`, próbka `poseidon_000001`
(T = 998,7 K, R_p = 0,979 R_jup, log g = 3,27), na POSEIDON **1.4.0 i 1.3.2** (wynik identyczny na obu —
patrz niżej). Etap 1 nie wymaga bazy opacity. Wersję generatora podaje `data/TauREx set/manifest.json`.

**Wyjaśnienie.** Wszystko, co Etap 1 potrafi sprawdzić, jest poprawne:

| sprawdzenie | wynik |
|---|---|
| mieszaniny w atmosferze vs żądane | `X` (7, 100, 1, 1); ślady −10,172 / −8,054 / −5,576 / −5,568 / −10,973 — **identyczne z etykietami** |
| tło H2/He | 0,850 / 0,150 — czyli `He_fraction = 0,17647` jako He/H2 daje zamierzone proporcje |
| profil temperatury | jedna unikalna wartość (izoterma), zgodnie z `PT_profile="isotherm"` |
| promień referencyjny | warstwa najbliższa 10 bar ma `r = 69 982,4 km` = **dokładnie** przekazane `R_p_ref` |
| monotoniczność | `P[0]=1e-6 bar` → `r=73 249,6 km`; `P[-1]=100 bar` → `r=69 554,5 km` — poprawnie |
| skala wysokości | mediana `H` = 200,46 km (zakres 190,48–211,25; 192,83 przy `P_ref` — `H` **nie** jest stałe pod izotermą, bo `g` maleje jak 1/r² na 18 skalach); rozciągłość 3695 km, czyli 18,43 H |
| **kierunek siatki ciśnień** | **bez znaczenia** — obie siatki dają bit-identyczny wynik; POSEIDON sortuje wewnętrznie |


Płaskie widmo ma dwie możliwe przyczyny, dające ten sam objaw z przeciwnych powodów: **opacity zerowe**
(atmosfera nie pochłania nic, więc światło blokuje wyłącznie ciało stałe planety) albo **opacity
nasycone** (atmosfera pochłania tak silnie, że τ=1 jest osiągane już w jej najwyższej, najcieńszej
warstwie). Test rozstrzyga między nimi,
sprawdzając, **któremu promieniowi fizycznemu odpowiada zapisana głębokość tranzytu** — bo obie
hipotezy przewidują inny promień efektywny. Promień wyliczony z zapisanej głębokości (73 268,3 km)
pasuje do promienia **szczytu atmosfery** (73 249,6 km, zgodność 99,95 %) i wyraźnie odbiega od dna
atmosfery oraz od `R_p_ref` (odpowiednio 90,1 % i 91,2 %). Zgodność ze szczytem, nie z dnem, wskazuje
na **nasycenie**, nie na brak opacity: światło jest blokowane najwyżej, jak to możliwe w tym modelu, co
oznacza nadmiar ekstynkcji, a nie jej brak.

Test rozróżniający, który wskazał przyczynę. Zapisana płaska głębokość dla tej próbki to
0,01698546. Promień, który ją implikuje, to **73 268,3 km**. Porównanie z trzema kandydatami:

| poziom | r [km] | (r/R_s)^2 | iloraz do zapisanej |
|---|---:|---:|---:|
| dno atmosfery (100 bar) | 69 554,5 | 0,01530721 | 0,901 |
| R_p_ref (10 bar) | 69 982,4 | 0,01549613 | 0,912 |
| **szczyt atmosfery (1e-6 bar)** | **73 249,6** | **0,01697681** | **0,9995** |

Zgodność do **0,05 %**; różnica 19 km to 0,1 skali wysokości.

Co to zmienia dla Etapu 2. Pytanie przestaje brzmieć "czy `read_opacities` zwróciło zera", a
zaczyna "**dlaczego ekstynkcja nasyca**". Kandydaci w kolejności prawdopodobieństwa: niezgodność bazy
opacity (`opacity_database="High-T"`, `database_version="1.3"`, przy czym bootstrap sprawdza obecność
bazy *standardowej*, więc mogła zostać wczytana niewłaściwa tabela), błąd jednostek w tablicach
przekrojów czynnych, albo zachowanie `opacity_sampling` poza siatką `T_fine`/`log_P_fine`.

DECYZJA PROJEKTOWA (2026-07-27): przyczyna zostaje zamknięta jako ZAWĘŻONA. Etap 2 wymaga 72,1 GB
danych wejściowych. Do celów raportu wystarcza to, co
ustalono: wrapper woła POSEIDON-a zgodnie z dokumentacją, atmosfera jest w pełni poprawna, a defekt
leży w torze opacity i objawia się **nasyceniem**, nie zerem. To uzasadnia wycofanie osi głównej i
zaplanowanie regeneracji; nie uzasadnia twierdzenia o konkretnej linii kodu. Etap 2 przechodzi do
backlogu jako warunek wstępny (regeneracja danych).

Wersja przypięta i zweryfikowana. Zbiór wygenerowano POSEIDON-em **1.3.2** (`manifest.json`). Diagnoza
Etapu 1 była pierwotnie wykonana na 1.4.0 (domyślna instalacja z `main`); powtórzona na przypiętym tagu
`v1.3.2` (`7077f1036e1aa46008efa262ba645dfe7ec1fc7e`) daje **bit-identyczne** wielkości geometrii i tę
samą przyczynę (`SATURATED`). Instalacja 1.3.2 wymagała dwóch dodatkowych zależności systemowych, obie
udokumentowane w bloku „komenda" niżej: `open-mpi` (biblioteka MPI, potrzebna przez `mpi4py`, którego
importuje `POSEIDON.core`) i `setuptools<81` (nowsze wersje usunęły `pkg_resources`, od którego zależy
`pysynphot`, importowany przez `POSEIDON.stellar`). Przypięcie 1.3.2 dla Etapu 2 (72,1 GB, backlog B1.1)
ma więc już potwierdzoną podstawę — geometria się nie zmienia między wersjami minor, więc pozostaje
tylko kwestia toru opacity, nie wersji biblioteki.

Literatura. arXiv:1409.2312 (Waldmann i in. 2014, TauREx I) opisuje, które elementy konfiguracji
forward-modelu decydują o amplitudzie cech — CIA, ciśnienie referencyjne, listy linii — i to jest
lista, którą Etap 2 musi przejść pozycja po pozycji. arXiv:1904.05356 (Welbanks & Madhusudhan 2019)
pokazuje, że wolno zamrozić `R_ref` **albo** `P_ref`, ale nie oba, i że CIA jest niezbędne; Etap 1
potwierdza, że u nas `R_p_ref` i `P_ref` są przekazane poprawnie (warstwa 10 bar odtwarza
`R_p_ref` dokładnie), więc kandydat „zła konwencja promienia referencyjnego" jest wykluczony i
zostaje sam tor przekrojów czynnych.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/d01_poseidon_diagnosis.py --stage 1`|
| komenda | POSEIDON-a nie ma w `.venv-qml` ani `.venv-cnn`, więc check wymaga osobnego interpretera i nie wchodzi do `run_all.SUITE`. Instalacja na przypiętej wersji generującej (1.3.2): `uv venv --python 3.11.9 <venv> && uv pip install 'POSEIDON @ git+https://github.com/MartianColonist/POSEIDON.git@v1.3.2' h5py pandas pyarrow open-mpi 'setuptools<81'` (`open-mpi` i `setuptools<81` konieczne — inaczej `RuntimeError: cannot load MPI library` / `ModuleNotFoundError: pkg_resources`). Uruchomienie: `MPLCONFIGDIR=/tmp/mpl <venv>/bin/python -u audit/d01_poseidon_diagnosis.py --stage 1` (`-u` obowiązkowe, inaczej segfault gubi stdout) |
| kryterium PASS | dokładne wywołanie z repo odtwarza widmo o tej samej zmienności relatywnej co tau (~1e-2). Etap 1 tego kryterium **nie może** spełnić, bo do widma potrzebna jest baza opacity. Status rekordu to jednak `FAIL`, nie `INFO`: `saturation_test` rozstrzyga znak defektu bez żadnych danych opacity, a nasycenie jest defektem |
| wartość oczekiwana | Wszystkie liczby geometrii i testu rozróżniającego cytowane w „Wyjaśnieniu" wyżej pochodzą z tego payloadu, zmierzone na 1.4.0 i potwierdzone bit-identycznie na 1.3.2. Dodatkowo: `input_data_root = null`, `opacity_files = []` (blocker Etapu 0 — Etap 2 potrzebuje zenodo 16107813, 72,1 GB); `grids_bit_identical = true`; `verdict_H1` niepotwierdzona. Rdzeń dowodu, `saturation_test`: `recorded_depth = 0,016985462978482246` (`n_unique_bins = 1`), `R_s_km = 562 182,28`, `implied_radius_km = 73 268,254`; ilorazy `atmosphere_bottom = 0,9012`, `R_p_ref = 0,9123`, `atmosphere_top = 0,9995` przy `tolerance = 5e-3` → `matching_candidates = [atmosphere_top]`, `verdict_opacity = SATURATED` |
| co by to obaliło | **przyczyna jest zawężona, nie ustalona**. Zawężenie („defekt leży w torze opacity i objawia się nasyceniem, nie zerem") upada, gdyby: odwrócenie siatki ciśnień zmieniało wynik (nie zmienia — bit-identyczny na obu wersjach), albo gdyby `(r_szczyt/R_s)²` nie zgadzało się z zapisaną głębokością do 0,05 %, albo gdyby Etap 2 pokazał poprawnie wczytane przekroje czynne i powierzchnię tau = 1 **nie** przy 1e-6 bar. W takim wypadku płaskość ma źródło poza torem opacity. Odwrotnie: **wskazanie konkretnej linii kodu wymaga Etapu 2** (72,1 GB, warunek wstępny regeneracji danych), dopóki go nie ma, każde twierdzenie o pojedynczej przyczynie w torze opacity jest nieuzasadnione i raport go nie stawia |
| rekord | 1.4.0: `reports/audit/20260727/d01_poseidon_diagnosis.json`; **1.3.2 (wersja generująca): `reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json`**, `timestamp_utc` 2026-07-29T17:39:54Z |

---

### K2(b). Drabina baseline'ów — skąd bierze się skill (dwa wyniki NA KORZYŚĆ projektu)

**Gdzie znaleziony.** `audit/a26_baseline_ladder.py` na trzech zbiorach (ADC2023 holdout n=4143,
crossgen tau_val n=4142, crossgen poseidon_test n=685); receptura ridge'a przepisana z
`data/crossgen_biosignatures/baseline_smoke.py`.

**Wyjaśnienie.** Cztery szczeble, każdy ignorujący ściśle mniej informacji, po dwóch
uczących się (ridge — ta sama receptura co w `baseline_smoke.py` — oraz gradient boosting),
12 000 wierszy treningowych.

Kolumnę `poseidon/test` należy czytać w świetle K1: te widma są stałe po długości fali, więc żaden
szczebel nie ma tam z czego się uczyć i wszystkie wartości poniżej zera są oczekiwane, nie diagnostyczne.

| szczebel | ADC holdout | skill | tau/val | skill | poseidon/test | skill |
|---|---:|---:|---:|---:|---:|---:|
| 0 — stała (średnia treningowa) | 1.4405 | 0.000 | 2.8853 | 0.000 | 2.8938 | 0.000 |
| 1 — **tylko aux, bez widma** | 1.4407 | **−0.000** | 2.8860 | **−0.000** | 2.8939 | −0.000 |
| 2 — **tylko widmo, bez aux** | **0.4409** | 0.694 | **1.6055** | 0.444 | 2.9498 | **−0.019** |
| 3 — aux + widmo | 0.4423 | 0.693 | 1.5964 | 0.447 | 2.9695 | **−0.026** |
| — raportowane modele: noquant / quantum | 0.2994 | 0.792 | 1.423 / 1.449 | 0.507 / 0.498 | 3.280 / 3.216 | −0.133 / −0.111 |

Wynik 2, również na korzyść: ExoBiome realnie bije kompetentny baseline nieneuronowy. Gradient
boosting na widmie daje 0.4409 na ADC holdout wobec 0.2994 ExoBiome — to przewaga **1,47×**, uzyskana
na tym samym wejściu i tym samym splicie, bez żadnej z pułapek z K3. Na tau/val analogicznie: 1.6055
wobec 1.423/1.449. To jest liczba, którą **wolno cytować** i która nie zależy od żadnego z zakwestionowanych
porównań.

Literatura. Nieneuronowy uczący się na widmach nie jest baseline'em ad hoc: arXiv:1806.03944
(Márquez-Neila i in. 2018) i arXiv:2405.02656 (Lueber i in. 2024, random forest / HELA) to uznane w
retrievalu atmosfer zastosowania lasów losowych, więc szczebel 2 tej drabiny jest tym samym typem
odniesienia, którego używa literatura — a nie „słabym baseline'em ustawionym po to, żeby przegrał".

Zgodność z K1, tym razem niezależna od sieci neuronowych. Na POSEIDON **ridge i GBM też mają ujemną
skill**: **−0,019** (ridge, tylko-widmo) i **−0,127** (GBM, tylko-widmo). Nie zawodzą więc akurat modele
neuronowe — zawodzi każda z trzech klas uczących się. Granica tego argumentu: wszystkie szczeble są
**uczone na `tau/train` i oceniane na POSEIDON** (`a26:364`), więc zerowy skill jest zgodny zarówno z
brakiem informacji w danych, jak i z brakiem transferu między generatorami; sam pomiar tych dwóch
możliwości nie rozdziela. Rozdziela je `a01` — pomiar na danych, bez udziału modelu. Drabina jest zatem
**zgodnością z przewidywaniem K1**, a nie jego niezależnym dowodem. To samo pokazuje zespołowy
`baseline_smoke.json` z bundla — patrz K1(b).


**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a26_baseline_ladder.py` |
| komenda | `./.venv-qml/bin/python audit/a26_baseline_ladder.py` |
| kryterium PASS | skill „tylko aux" < 20 % skillu „aux + widmo" |
| wartość oczekiwana | Status `WARN` (był `PASS`, dopóki szczebel `poseidon_test` z zerowym skillem milcząco go wspierał — patrz `NAPRAWY_KODU`). `skill_share_of_aux` = **`−0.0019`** (ADC) i **`−0.0041`** (tau_val), czyli udział tabeli aux w skillu wynosi −0,19 % / −0,41 %; `skill_share_of_spectrum_only` = `1.0014` (ADC) i `0.9930` (tau_val), więc szczebel „tylko widmo" **dorównuje lub bije** szczebel „aux + widmo". Na `poseidon_test` `skill_share_of_aux = null`, bo mianownik jest ujemny. Szczebel 2: ADC gbm `0.4409` (skill `0.6939`) wobec ridge `1.0174` (`0.2937`); tau_val gbm `1.6055` (`0.4436`); poseidon_test ridge `2.9498` (`−0.0194`) i gbm `3.2623` (`−0.1274`). Szczebel 0: `1.4405` / `2.8853` / `2.8938`. `n_train = 12000` na każdym zbiorze |
| co by to obaliło | wynik 1 (aux nie nosi skillu) upada, gdy `skill_share_of_aux` ≥ 0,20 — czyli gdy szczebel 1 dorównuje jednej piątej szczebla 3; wtedy hipoteza wycieku przez tabelę pomocniczą wraca. Wynik 2 (przewaga 1,47×) upada, gdy GBM na widmie zejdzie do ≤ 0.2994 na ADC holdout przy tym samym splicie i tym samym budżecie — na przykład po tuningu, którego drabina celowo nie robi; przewaga jest więc dolnym oszacowaniem tylko wobec **nietuningowanego** GBM. Trzecie potwierdzenie K1 upada, gdyby ridge albo GBM osiągnęły na POSEIDON skill > 0 — to jednak nie obaliłoby samego K1, który stoi na `a01`, a nie na tej drabinie |
| rekord | `reports/audit/20260729/a26_baseline_ladder.json` |

---

### K3. Benchmark vs SOTA porównuje różne wejścia

**Gdzie znaleziony.** Sporne twierdzenie — publikowana przewaga 1,85× — stoi w raportach porównawczych
ADC; split potwierdza `saved_split_manifest.json`. Rozjazd konwencji jest wypalony w kodzie treningowym
na main, w pięciu miejscach opisanych niżej. Kluczowa obserwacja: **po stronie NSF przełącznik szumu
istnieje i jest zaszyty na `True` w każdym wywołaniu, a po stronie ExoBiome takiego przełącznika nie ma
w ogóle** — nie da się więc „wyłączyć szumu" po jednej stronie, bo asymetria jest w architekturze
skryptów, nie w konfiguracji przebiegu.

**(a) NSF — wtrysk szumu i jego pięć wywołań.** Sam wtrysk, `models/adc_winner_on_ariel/preprocessing.py:142-151`:

```python
    sample_noise: bool,
    noise_generator: torch.Generator | None = None,
) -> torch.Tensor:
    dtype = spectra.dtype
    device = spectra.device

    if sample_noise:
        sampled_spectra = torch.normal(mean=spectra, std=noise, generator=noise_generator)
    else:
        sampled_spectra = spectra
```

`models/adc_winner_on_ariel/train.py` woła to z `sample_noise=True` **bez żadnej flagi konfiguracyjnej**
w pięciu punktach: `:161` (pętla treningowa), `:181` (walidacyjny NLL), `:219` (walidacyjny mRMSE per
epoka), `:253` (finalna walidacja), `:267` (finalny holdout). Pętla treningowa, `:156-163`:

```python
            context = build_context_batch(
                data.train,
                batch_rows,
                data.runtime_scalers or data.scalers,
                device=device,
                sample_noise=True,
                noise_generator=noise_generator,
            )
```

Rozstrzygające jest jednak `:257-272`, bo tam konwencja zaszumiona trafia **do artefaktu**, z którego
`a02` czyta 0.5523:

```python
        holdout_metrics = evaluate_point_metric(
            model,
            data.holdout,
            …
            sample_noise=True,
            noise_seed=metric_noise_seed,
            log_prefix="holdout final",
        )
        save_metrics(run_dir / "validation_metrics.json", val_metrics)
        save_metrics(run_dir / "holdout_metrics.json", holdout_metrics)
```

Czyli szum nie jest dodawany dopiero przez `scripts/reeval_sota.py` — reeval tylko powtarza konwencję,
w której powstał `holdout_metrics.json`.

**(b) ExoBiome — nie ma czego wyłączyć.** `instrument_noise` wchodzi jako **statyczny kanał wejściowy**,
nie jako σ do losowania — `models/ariel_exobiome/constants.py:44-47`:

```python
SAMPLE_SPECTRAL_CHANNELS = [
    "instrument_spectrum",
    "instrument_noise",
]
```

Model więc *widzi* wektor szumu (`MODEL_SPECTRAL_CHANNELS`, `:54-59`), ale samo widmo nigdy nie jest
perturbowane. Sygnatura `evaluate_labeled_split` w `models/ariel_exobiome/training.py:238-246` nie ma
parametru szumu, a `gather_labeled_batch` (`:255`) podaje zapisane spektra prosto do `model(...)`:

```python
def evaluate_labeled_split(
    model: HybridArielRegressor,
    split: LabeledSplit,
    target_scaler,
    batch_size: int,
    loss_fn: nn.Module,
    enable_quantum: bool = True,
    quantum_scale: float = 1.0,
) -> dict[str, Any]:
```

Potwierdzenie negatywne: w `models/ariel_exobiome/dataset.py` szukanie `noise|rng|random|jitter|perturb`
trafia wyłącznie w `random_state=seed` (`:467`) i `random_state=seed + 1` (`:478`), czyli w podział
zbioru. Ani jednej ścieżki próbkującej szum.

**(c) Kryterium stopu ≠ metryka raportowana** (kodowe potwierdzenie „niedotrenowanego baseline'u"). W tej
samej pętli `models/adc_winner_on_ariel/train.py` licznik cierpliwości rośnie **wyłącznie** w gałęzi NLL
(`:197-202`):

```python
        if val_nll < state.best_val_nll:
            state.best_val_nll = val_nll
            state.epochs_since_improvement = 0
            save_checkpoint(run_dir / "best_model_by_nll.pt", …)
        else:
            state.epochs_since_improvement += 1
```

`best_model_by_mrmse.pt` powstaje osobno i **tylko** wewnątrz `if metric_every > 0 and epoch %
metric_every == 0` (`:228-230`), czyli mRMSE jest mierzone co kilka epok:

```python
            if metrics["rmse_mean"] < state.best_val_rmse:
                state.best_val_rmse = metrics["rmse_mean"]
                save_checkpoint(run_dir / "best_model_by_mrmse.pt", …)
```

a przerwanie następuje na cierpliwości NLL (`:236-238`):

```python
        if state.epochs_since_improvement >= patience:
            print(f"Early stopping after epoch {epoch} due to validation NLL patience.", flush=True)
            break
```

Postęp na mRMSE nie może więc utrzymać treningu przy życiu, mimo że to `best_model_by_mrmse.pt` jest
potem ładowany (`:240-242`) i z niego biorą się raportowane liczby. To jest mechanizm stojący za stopem
na epoce 79/300 opisanym niżej.

**(d) Ta sama asymetria obowiązuje w ramieniu cross-generator**. Identyczny kod i te same numery linii w przeniesionych
bliźniakach: `models/ariel_winner_on_taurex/preprocessing.py:148-149` oraz `train.py:161,181,219,253,267`
i `evaluate.py:31`; tak samo `models/ariel_winner_nf/` i `models/ariel_winner_trace_nf/ariel_winner_nf/`.
Po stronie przeciwnej `models/taurex_exobiome_without_quant/` nie ma **ani jednego** dopasowania na
`torch.normal|np.random.normal|randn|default_rng|sample_noise`; jego `augment_sample_spectra`
(`dataset.py:449-461`) skleja wyłącznie kanały pochodne, deterministycznie:

```python
    spectrum = normalized[:, 0, :]
    noise = normalized[:, 1, :]
    gradient = np.diff(spectrum, axis=1, prepend=spectrum[:, :1])
    curvature = np.diff(gradient, axis=1, prepend=gradient[:, :1])
    signal_to_noise = spectrum / np.clip(noise, 1.0e-6, None)
    augmented = np.stack([spectrum, noise, gradient, curvature, signal_to_noise], axis=1).astype(np.float32)
```

**(e) Oficjalny baseline ADC używa trzeciej konwencji** — augmentacji **treningu** replikami szumu, a nie
wstrzyknięcia przy ewaluacji. `models/adc_baseline/helper.py:29-33`:

```python
def augment_data(arr, noise, repeat=10):
    noise_profile = np.random.normal(loc=0, scale=noise, size=(repeat,arr.shape[0], arr.shape[1]))
    ## produce noised version of the spectra
    aug_arr = arr[np.newaxis, ...] + noise_profile
    return aug_arr
```

Wołane przez `models/adc_baseline/preprocessing.py:4-5` i `models/adc_baseline/run_baseline.py:83-84`.
Warto to odnotować, bo liczba CNN 0.6500 stoi w tej samej tabeli co pozostałe dwie.

**Wyjaśnienie.** Ten sam checkpoint, ten sam holdout (4143 wiersze, split potwierdzony przez `saved_split_manifest.json`):

| wejście | ExoBiome (258 688 par.) | winner-style NSF (10 771 200 par.) |
|---|---:|---:|
| widmo jak w pliku | **0.298693** (skala 1.0) · 0.295552 (0.5) · 0.302409 (kwant OFF) **Trenowany na takich danych**| **0.403603** (mean) · **0.384621** (median) |
| widmo + N(0,σ), 3 seedy | **0.966512 ± 0.006572** | **0.548927 ± 0.000863** (mean) · 0.556800 ± 0.000611 (median) ← *stąd raportowane 0.5523* **Trenowany na takich danych**|

- Publikowana przewaga **1,85×** to porównanie najlepszej komórki ExoBiome z najgorszą komórką baseline'u.
- Przy zrównanym wejściu: **1,29×** na ramieniu `median` — tym, które zespół opublikował (`models/adc_winner_on_ariel/trained_run/holdout_metrics.json: point_estimate = "median"`; klucz nazywa się myląco `rmse_mean`, ale to średnia po gazach, nie ramię `mean`) — oraz 1,35× na ramieniu `mean`. Porownanie jest więc zrównane **podwójnie**: to samo wejscie i ten sam estymator punktowy.
- Degradacja pod szumem: ExoBiome **×3,24**, flow **×1,36** (mean) / **×1,45** (median). Hipoteza „odporność
  na szum dzięki kwantom" nie jest tym poparta — pierwszy dostępny pomiar jest **przeciw** niej. Czego ten
  pomiar **nie** rozstrzyga: ExoBiome nie widział szumu w treningu ani raz, a flow widział świeżą realizację
  co epokę, więc różnica degradacji mierzy **różnicę reżimów treningowych**, nie odporność architektury.
  Rozstrzygnięcie wymagałoby dotrenowania ExoBiome z tą samą augmentacją.
- Ubocznie, i to jest ustalenie o **protokole ewaluacji modelu odniesienia**, nie o naszym kodzie:
  posterior flow jest losowany bez zaseedowania **po obu stronach**. W kodzie zespołu
  `models/adc_winner_on_ariel/model.py:54` woła `flow(context).sample((num_samples,))` bez generatora,
  czyli z globalnego RNG torcha; `evaluate.py:46-47` seeduje `torch.Generator`, ale przekazuje go
  **wyłącznie do szumu wejściowego**, a nie do `model.sample(...)` w l. 63. Wynika z tego, że
  **opublikowane `0.5523` również nie jest bitowo odtwarzalne**.

**Ile szumu jest w widmach ADC?** Dokumentacja zbioru odpowiada na to wprost: widma są idealne, szum nie jest do nich wliczony, a tablica σ jest dostarczana osobno (arXiv:2309.09337, §1.2 Noise; potwierdzone w publicznym repozytorium AstroAI-CfA). Nasz własny test rozrzutu bin-do-binu (a09, status FAIL) jest zgodny z kierunkiem „mniej niż pełna σ", ale nie rozstrzyga tego niezależnie — brakuje kalibracji na tym samym binowaniu.

Test mierzy statystykę `R = sqrt( Σd²ᵢ / Σ(σ²ᵢ₋₁ + 4σ²ᵢ + σ²ᵢ₊₁) )`, gdzie `dᵢ` jest drugą różnicą widma
po długości fali. Dla szumu i.i.d. na poziomie dokładnie σ zachodzi `E[R] = 1`; dla widma gładkiego bez
szumu `R` zbiega do zera. Dwa pierwsze wiersze są kalibracją na zbiorze, w którym poziom szumu jest znany:

| zbiór | p5 | mediana | p95 | frakcja < 0.9 |
|---|---:|---:|---:|---:|
| tau, `transit_depth_noiseless` (wiadomo: brak szumu) | 0.048 | **0.370** | 4.240 | 0.732 |
| tau, `transit_depth_noisy` (wiadomo: dokładnie 1.0 σ) | 0.929 | **1.093** | 4.315 | 0.023 |
| ADC `instrument_spectrum` (pytanie otwarte, W14) | 0.319 | **0.757** | 2.108 | 0.627 |

**(f) Trzy osie rozjazdu, nie jedna — wektor celu i źródło nadzoru.** Ta sekcja mierzy oś pierwszą.
Pozostałe dwie nie są w niej zmierzone, a druga z nich dotyczy **estymandu**, nie konfiguracji:

| oś | ExoBiome | model porównawczy w repo (`adc_winner_on_ariel`) | zwycięskie zgłoszenie ADC2023 |
|---|---|---|---|
| konwencja wejścia | widmo jak w pliku | widmo + N(0,σ) w treningu **i** ewaluacji | widma idealne |
| wektor celu | 5 gazów | 5 gazów (`constants.py:19-25`) | **7** parametrów (R_p, T + 5 gazów) |
| źródło nadzoru | parametry wejściowe FM | parametry wejściowe FM | **ważone próbki nested samplingu** (`Tracedata.hdf5`) |

W taksonomii pracy zwycięzców (arXiv:2309.09337, Fig. 4) nasza reimplementacja odpowiada modelowi
**„alternative noised"**, score **577,32** — nie modelowi zwycięskiemu, score **688,13** (trzeci wariant,
„alternative ideal", dostał 457,83). Autorzy sami argumentują, że model noised jest lepszy naukowo mimo
niższego score'u (§3.1, §4.1 p. 2), więc **wybór baseline'u nie jest błędem** — błędem jest etykieta.
Ścieżka faktycznej rodziny zwycięskiej **istnieje w repo i nigdy nie została wytrenowana**:
`models/ariel_winner_trace_nf/` deklaruje cele z `Tracedata.hdf5` i 7 parametrów (README + `constants.py`),
a `best_independent_bundle.pt` nie istnieje w żadnym worktree.

Brzmienie dopuszczalne: „reimplementacja podejścia zwycięzcy ADC2023 — niezależne marginalne, 5 gazów, cele
FM, augmentacja szumem". Niedopuszczalne: „zwycięzca ADC2023", „SOTA" bez kwalifikatora.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a03_input_convention.py` (główny); wspierające: `audit/a09_noise_realization.py` — ile szumu jest w widmach ADC, oraz `audit/a05_training_completeness.py` — czy baseline jest dotrenowany na metryce porównania. Oba mają status `FAIL` i własne rekordy |
| komenda | `./.venv-qml/bin/python audit/a03_input_convention.py --split holdout --seeds 42,1,2` |
| kryterium PASS | `sign(ExoBiome − baseline)` niezmienny wobec konwencji wejścia |
| wartość oczekiwana | Status `FAIL`, `sign_flips = true`. ExoBiome: czysty `0.298693` (skala 1.0, raportowana — pełny sweep skal w K4), zaszumiony `0.966512 ± 0.006572` (3 seedy) → `degradation_factor = 3.236`. NSF (`trained_with_noise_augmentation = true`, wartość zaszyta w `a03:118`, potwierdzona lekturą `preprocessing.py:146-149`, `train.py:161` i in.): czysty `0.384621` (ramię **publikowane**, `median`), zaszumiony `0.556800 ± 0.000611` → `degradation_factor = 1.448`. Porównanie na ramieniu publikowanym: `claimed_ratio = 1.845`, `ratio_clean = 1.288`, `ratio_noised = 0.576` (kierunek odwrócony). Liczby NSF niepowtarzalne w 3. cyfrze znaczącej (§1a, niezaseedowany `model.sample()`) — cytować z podaniem przebiegu, tu `20260729`. Wspierające: `a09`, `a05` (§K5) |
| rekord | `reports/audit/20260729/a03_input_convention.json` (`timestamp_utc` = 2026-07-29T12:48Z; wspierające: `a09_noise_realization.json`, `a05_training_completeness.json` z tego samego przebiegu) |

---

### K4. Raportowane metryki przy nieważonym `quantum_scale`

**Gdzie znaleziony.** `models/ariel_exobiome/training.py:771-772` — `validation_metrics` i
`holdout_metrics`, czyli `models/ariel_exobiome/training.py` woła `evaluate_labeled_split` dla finalnych
metryk **bez** argumentu `quantum_scale`, więc bierze default
`quantum_scale: float = 1.0` z sygnatury (`models/ariel_exobiome/training.py:245`), podczas gdy selekcja
epoki w tej samej pętli podaje skalę jawnie (`models/ariel_exobiome/training.py:565-572`,
`quantum_scale=quantum_scale`), czyli szła przy ramped scale.

**Wyjaśnienie.** Sweep na tym samym checkpoincie (`artifacts/ariel_quantum_best_v4_epoch6/best_model.pt`)
i tym samym splicie:

Wszystkie wiersze to **ewaluacje tych samych wag** przy różnym mnożniku `quantum_scale` (mnożnik działa
w czasie inferencji, `model.py:327`), a nie osobno trenowane modele:

| skala przy ewaluacji | holdout mRMSE | validation mRMSE |
|---|---:|---:|
| 0.0 (kwant OFF) | 0.302409 | 0.299271 |
| 0.25 | 0.297749 | 0.294569 |
| **0.5 — skala, przy której checkpoint był trenowany i wybrany (epoka 6)** | 0.295552 | 0.292237 |
| 0.6667 — najwyższa skala osiągnięta w treningu (epoka 8) | 0.295487 | 0.292028 |
| **1.0 — default sygnatury funkcji, w treningu nieosiągnięty** | **0.298693** ← `mac_holdout_metrics.json` | **0.294821** |

Zacommitowane `holdout_metrics.json = 0.299376` i `validation_metrics.json = 0.293614` pochodzą z tej samej
ewaluacji przy skali 1.0 wykonanej na CUDA + AMP.

Konsekwencje: (a) model nigdy nie był trenowany przy skali 1.0, przez co nie mógł dostosować wag do tej skali (b) flagowa liczba jest z punktu pracy, który nigdy nie był walidowany, jest **gorsza** od
wybranego (0.298693 vs 0.295552) i nie jest optimum sweepu (`best_scale = 0.6667`, `reported_is_best = False`);
(c) wkład całej ścieżki kwantowej wynosi **0.003716** przy raportowanym punkcie i **0.006857** przy skali
selekcji (d) `reports/taurex_noquant_mac_20260312_133054_audit.md:10` (opis buga; naprawa jest w kodzie — `models/taurex_exobiome_without_quant/training.py:863` przepisuje `best_refinement_scale` do ewaluacji koncowej)
**opisuje ten bug i naprawia go w kontroli klasycznej** (`final_refinement_scale = best_refinement_scale`),
więc każde head-to-head ocenia kwant przy skali nieważonej, a kontrolę przy wybranej — asymetria pomiaru.

**Potwierdzenie z historii treningu, niezależne od sweepu.** Skala jest **mnożnikiem stosowanym w czasie
inferencji** — `models/ariel_exobiome/model.py:327` zwraca `classical_pred + quantum_scale · gate ·
quantum_correction`, więc dowolną wartość można podać dowolnym wagom. Wiersze tabeli wyżej są zatem
**ewaluacjami tego samego checkpointu** przy pięciu różnych mnożnikach, a nie pięcioma wytrenowanymi
modelami; stąd w tabeli istnieje wiersz `1.0`.

W treningu ta sama skala narasta według harmonogramu: `config.json` deklaruje `quantum_ramp_epochs = 12`,
a `resolve_quantum_scale` (`training.py:335-341`) liczy ją jako `min(1.0, active_epoch /
quantum_ramp_epochs)`, więc wartość 1.0 wymagałaby **12 epok**. Run zakończył się na epoce 8 przy skali
`0.6667` (`training_state.json`), a wysłany checkpoint pochodzi z epoki 6, gdzie skala wynosi **dokładnie
0,5** (`history.csv`).

Rozróżnienie jest więc następujące i to ono jest treścią K4: **model nigdy nie był trenowany przy skali
1.0**, a mimo to flagowe metryki policzono, ustawiając mnożnik na 1.0 przy ewaluacji. Zarzut nie brzmi
„ta liczba nie mogła powstać", lecz „powstała w punkcie pracy, w którym model nie był ani trenowany, ani
walidowany", przy czym punkt walidowany (0,5) daje wynik **lepszy** (0.295552 wobec 0.298693). To zamyka
warunek `max(quantum_scale) ≥ 1.0` z tabeli K5 rekordem treningu, a nie inferencją ze sweepu.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a04_quantum_scale_provenance.py` |
| komenda | `./.venv-qml/bin/python audit/a04_quantum_scale_provenance.py` |
| kryterium PASS | skala odtwarzająca opublikowane liczby == skala z chwili selekcji odczytana z `history.csv` |
| wartość oczekiwana | `status = FAIL`; `best_epoch = 6`, `selection_time_quantum_scale = 0.5`, `selection_time_val_mrmse = 0.290811`; `scales_matching_a_published_number = ["1.000000"]` na obu splitach (holdout `n_rows = 4143`, validation `n_rows = 4142`); `matches_mac_reeval = true` wyłącznie przy 1.0; `gate_off_mrmse = 0.302409` (holdout) i `0.299271` (validation); `quantum_pathway_contribution_at_scale_1 = 0.003716` (holdout) / `0.004450` (validation); `..._at_selection_scale = 0.006857` (holdout) / `0.007034` (validation); `best_scale_on_this_split = 0.666667`, `reported_is_best = false` |
| co by to obaliło | gdyby `training.py:771-772` przekazywało `quantum_scale=quantum_scale` (jak linia 565-572), albo gdyby sweep przy 0.5 dawał `matches_reported`/`matches_mac_reeval = true`. Osobno: `reported_is_best = true` (albo `best_scale_on_this_split = 1.000000`) usuwa część (a) zarzutu, bo raportowany punkt pracy byłby optimum sweepu, choć nadal niewalidowanym |
| rekord | `reports/audit/20260729/a04_quantum_scale_provenance.json` |

### K5. Flagowy checkpoint z runu przerwanego przy odmrożeniu backbone'u

**Gdzie znaleziony.** `artifacts/ariel_quantum_best_v4_epoch6/history.csv` (razem z `config.json` i
`training_state.json` tego samego artefaktu). Twierdzenie sporne to sama prezentacja tego checkpointu jako
modelu skończonego w `artifacts/ariel_quantum_best_v4_epoch6/README.md` i we flagowych tabelach opartych na
`holdout_metrics.json`.

**Wyjaśnienie.** `history.csv`:

| epoka | val mRMSE | quantum_scale | backbone_frozen |
|---:|---:|---:|---:|
| 1 | 0.29332 | 0.083 | 1 |
| 5 | 0.29128 | 0.417 | 1 |
| **6** | **0.29081** | **0.500** | 1 |
| 7 | 0.32866 | 0.583 | 0 |
| 8 | 0.32182 | 0.667 | 0 |

Run zakończony na epoce 8 z `max_epochs=30`, bez wyczerpania `early_stop_patience=8`, co oznaczałoby stop ręczny.
Najlepsza epoka to dokładnie ostatnia epoka fazy zamrożonego backbone'u, a po odmrożeniu walidacja rośnie
o 13 %. `quantum_scale` nigdy nie przekroczył 0.667, więc wersji architektury przy skali 1.0 nikt nie
wytrenował — a metryki są raportowane właśnie przy 1.0. Checkpoint stage-1, od którego stage-2 startuje
(`/home/iwo/hack4sages-crossgen/outputs/ariel_quantum_two_stage_v2_20260311_175726/stage1_classical/best_model.pt`),
**nie istnieje w repo**, więc całkowity budżet treningowy jest nieudokumentowany i żadne stwierdzenie
„compute-matched" nie jest możliwe w żadną stronę.

**Zarzut, trzy osobne zdania, i pierwsze jest na korzyść projektu.** Kolumna

1. **Faza adaptera kwantowego wykonała się w całości i zgodnie z projektem.** `quantum_backbone_freeze_epochs
   = 6` jest wartością zadeklarowaną w konfiguracji, a nie skutkiem przerwania: epoki 1–6 miały backbone
   zamrożony celowo, żeby uczył się wyłącznie adapter. W każdej z tych sześciu epok walidacja się poprawiała
   (0,29333 → 0,29081, monotonicznie). Zarzut „gałąź kwantowa nie zdążyła się wytrenować" jest więc
   wykluczony.
2. **Faza wspólna praktycznie się nie odbyła, a wysłany checkpoint jest sprzed niej.** Backbone odmarza na
   epoce 7; run kończy się na 8, czyli faza wspólna dostała **2 epoki z ~24 zaplanowanych** (30 − 6).
   `best_epoch = 6` to **ostatnia epoka z zamrożonym backbone'em**, więc publikowany model to klasyczny
   backbone ze stage-1 **nietknięty w stage-2** plus adapter kwantowy przy skali 0,5. Faza wspólna nie
   wniosła do wysłanych wag ani jednego kroku gradientu.
3. **Wzrost na epoce 7 jest skutkiem odmrożenia.** Skok val 0,29081 → 0,32866
   (+13,0 %) pokrywa się **dokładnie** z przejściem `backbone_frozen 1 → 0`.

**Stop był ręczny:** `best_epoch = 6`, `current_epoch = 8`, `early_stop_patience = 8`
(`training_state.json`), czyli wykorzystane **2 z 8** epok cierpliwości. Harmonogram runu go nie zatrzymał.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a05_training_completeness.py` |
| komenda | `./.venv-qml/bin/python audit/a05_training_completeness.py` |
| kryterium PASS | run zakończony własną regułą ∧ raportowana metryka na plateau ∧ metryka selekcji == metryka raportowana ∧ selekcja na pełnym splicie |
| wartość oczekiwana | `status = FAIL`, `n_runs_with_issues = 4`. Dla `exobiome_stage2_v4`: `epochs_run = 8`, `max_epochs = 30`, `best_epoch = 6`, `early_stop_patience = 8`; `val_trajectory = [0.293325, 0.292385, 0.291854, 0.291571, 0.291279, 0.290811, 0.328656, 0.321821]`; `quantum_scale_trajectory` od 0.083333 do 0.666667; `backbone_frozen_trajectory = [1,1,1,1,1,1,0,0]`; `selection_metric = "val mRMSE (full split, every epoch)"` vs `reported_metric = "val/holdout mRMSE at quantum_scale=1.0"` |
| co by to obaliło | `epochs_run == max_epochs` (czyli 30) albo `epochs_run − best_epoch ≥ early_stop_patience` (≥ 8) - run zakończył się legalnym early stopem, nie ręcznie. Dalej: `best_epoch > quantum_backbone_freeze_epochs`, teza „model nigdy nie trenował z odmrożonym backbonem" upada; `val[best] ≤ val[best−1]` - słowo „dywergencja" jest nadinterpretacją; `max(quantum_scale) ≥ 1.0` → zarzut „wersji przy skali 1.0 nikt nie wytrenował" upada; pojawienie się checkpointu stage-1 w repo - budżet treningowy staje się udokumentowany |
| rekord | `reports/audit/20260729/a05_training_completeness.json` |

Niezależnie od checku, ustalenie ma **drugą ścieżkę odtworzenia**: one-liner, który czyta wyłącznie surowe
artefakty (`history.csv`, `config.json`, `training_state.json`), bez `import audit_lib`, więc weryfikuje samo
ustalenie, a nie kod audytu.

```bash
cd ../hack4sages && ./.venv-qml/bin/python -c "
import json, pandas as pd
A='artifacts/ariel_quantum_best_v4_epoch6/'
h=pd.read_csv(A+'history.csv'); c=json.load(open(A+'config.json')); s=json.load(open(A+'training_state.json'))
print(h[['epoch','val_rmse_mean','quantum_scale','backbone_frozen']].to_string(index=False))
last=int(h.epoch.max()); best=int(s['best_epoch']); v=h.val_rmse_mean.values
print(f\"T1 stop reczny? ostatnia={last}<max={c['max_epochs']}; od najlepszej={last-best}<patience={c['early_stop_patience']}\")
print(f\"T2 best_epoch({best}) == freeze_epochs({c['quantum_backbone_freeze_epochs']})\")
print(f'T3 po odmrozeniu: {v[best-1]:.5f} -> {v[best]:.5f} ({100*(v[best]/v[best-1]-1):+.1f}%)')
print(f'T4 max quantum_scale = {h.quantum_scale.max():.4f}')"
```

| test | oczekiwane | co by obaliło K5 |
|---|---|---|
| T1 | ostatnia 8 < max 30; od najlepszej 2 < patience 8 → **stop ręczny** | `last == 30` albo `last − best ≥ 8`, early stop|
| T2 | `best_epoch = 6 == freeze_epochs = 6` | best po odmrożeniu |
| T3 | 0,29081 → 0,32866, **+13,0 %** | `val[best] ≤ val[best−1]` |
| T4 | max skali **0,6667** | `max ≥ 1,0` upada "wersji przy skali 1.0 nikt nie wytrenował" |

### K6. „Wkład kwantowy" nieodseparowalny od parametrów klasycznych

**Gdzie znaleziony.** `artifacts/ariel_quantum_best_v4_epoch6/best_model.pt` (`model_state_dict`) w zestawieniu
z `models/ariel_exobiome/model.py` — moduły `projector`, `quantum_block`, `quantum_head`, `quantum_gate`
składają się na to, co dokumentacja nazywa „ścieżką kwantową".

**Wyjaśnienie.** Policzone z wag:

| moduł | parametry |
|---|---:|
| spectral_encoder | 104 321 |
| aux_encoder | 1 344 |
| fusion_encoder | 33 280 |
| classical_head (256→192→5) | 50 309 |
| **projector (128→128→8)** | **17 560** |
| **quantum_block.weights** | **24** |
| **quantum_head (264→192→5)** | **51 845** |
| **quantum_gate** | **5** |
| **razem** | **258 688** |

- obwód = **0,0093 %** modelu (dokładnie **0,009278 %**); 1 warstwa wariacyjna
  (`depth=2 → num_blocks=1`, `3·n_qubits·num_blocks`)
- ścieżka „kwantowa" to **69 434** parametrów, z czego **24** kwantowe i **69 410 klasycznych**, czyli
  **+36,7 %** (dokładnie **+36,68 %**) względem modelu classical-only (189 254). Ablacja on/off **nie jest**
  param-matched

„Korekta kwantowa" jest drugą pełną głową klasyczną z 8 dodatkowymi cechami. W rozbiciu modułów zwróć uwagę
na `quantum_head = 51 845` wobec `classical_head = 50 309`: jest to po prostu druga, większa głowa
klasyczna. Test ON/OFF mierzy „druga głowa włączona/wyłączona", nie „kwant włączony/wyłączony". Stąd wniosek,
który **nie zależy od żadnego eksperymentu**: przy tej architekturze wynik zerowy jest **nieinformatywny**
(obwód nie miał jak zadziałać: jego wyjście zajmuje **8 z 264** wymiarów wejścia głowy i jest mnożone przez
`|tanh(gate)|·scale ∈ [0.0178, 0.0292]` — obie liczby są w payloadzie `a06`), a wynik dodatni
**nieatrybuowalny** (**+36,68 %** dodatkowej pojemności klasycznej jest
wystarczającym wyjaśnieniem alternatywnym). Więcej seedów tego nie naprawia.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a06_param_accounting.py` |
| komenda | `./.venv-qml/bin/python audit/a06_param_accounting.py` |
| kryterium PASS | gałąź kwantowa dokłada < 1 % dodatkowych parametrów klasycznych względem modelu classical-only |
| wartość oczekiwana | `status = FAIL`; `total_params = 258688`, `quantum_pathway_params = 69434`, `circuit_params = 24`, `extra_classical_params_added_by_the_branch = 69410`, `classical_only_params = 189254`; `circuit_share_of_model = 9.277585e-05` (0,009278 %), `extra_classical_over_classical_only = 0.366756` (+36,68 %); `quantum_head_input_dim = 264`, `quantum_features_share_of_head_input = 0.030303` (3,03 %); `variational_layers = 1`, `circuit_formula = "3 * n_qubits * (depth // 2)"`; `effective_multiplier_at_scale_0.5 = [0.017765, 0.029234]` |
| co by to obaliło | `extra_classical_over_classical_only < 1 %`, ablacja on/off byłaby param-matched.`circuit_params ≠ 24` lub kształt `quantum_block.weights` niezgodny z `3·8·(2//2)` |
| rekord | `reports/audit/20260729/a06_param_accounting.json` |

Ustalenie ma **drugą ścieżkę odtworzenia**: one-liner poniżej czyta wyłącznie surowe artefakty
(`best_model.pt`, `config.json`), bez `import audit_lib`, więc weryfikuje samo ustalenie, a nie kod audytu.


```bash
cd ../hack4sages && ./.venv-qml/bin/python -c "
import torch, json
A='artifacts/ariel_quantum_best_v4_epoch6/'
sd=torch.load(A+'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
c=json.load(open(A+'config.json')); g={}
for k,v in sd.items(): g[k.split('.')[0]]=g.get(k.split('.')[0],0)+v.numel()
tot=sum(g.values()); q=sum(g.get(k,0) for k in ('projector','quantum_block','quantum_head','quantum_gate'))
circ=sd['quantum_block.weights'].numel(); W=sd['quantum_head.net.0.weight']
[print(f'   {k:18}{v:>8}') for k,v in sorted(g.items(),key=lambda x:-x[1])]
print(f\"T1 wzor 3*{c['qnn_qubits']}*{c['qnn_depth']//2}={3*c['qnn_qubits']*(c['qnn_depth']//2)} vs ksztalt {tuple(sd['quantum_block.weights'].shape)}\")
print(f'T2 udzial obwodu {circ}/{tot} = {circ/tot:.6%}')
print(f'T3 sciezka kwantowa {q}, kwantowe {circ}, KLASYCZNE {q-circ}')
print(f'T4 classical-only {tot-q}, branch doklada +{(q-circ)/(tot-q):.2%}')
print(f\"T5 wejscie quantum_head {W.shape[1]} wym., z obwodu {c['qnn_qubits']} = {c['qnn_qubits']/W.shape[1]:.2%}\")"
```

| test | oczekiwane | co by obaliło K6 |
|---|---|---|
| T1 | `3·8·(2//2) = 24`, kształt `(24,)` | inna liczba|
| T2 | **0,009278 %** | — (opisowe) |
| T3 | 69 434 / 24 / **69 410 klasycznych** | — to jest treść ustalenia |
| T4 | classical-only 189 254; branch **+36,68 %** | `< 1 %` → ablacja on/off byłaby param-matched |
| T5 | **8 z 264 = 3,03 %** | wejście ≈ `n_qubits`, głowa nie byłaby full-context, zarzut osłabiony |

### K7. „Model sam wyciszył kwant" — twierdzenie odwrócone względem kodu

**Gdzie znaleziony.** `models/ariel_exobiome/model.py:260-262` (równoważnie
`models/taurex_exobiome/model.py:268-270`, `models/five_qubit_exobiome/model.py:261`) —
`self.quantum_gate = nn.Parameter(torch.zeros(len(TARGET_COLUMNS)))`.

**Wyjaśnienie.** `taurex_exobiome/model.py:268-270` = `ariel_exobiome/model.py:260-262`:
`self.quantum_gate = nn.Parameter(torch.zeros(len(TARGET_COLUMNS)))`.

| gaz | `tanh(gate)` |
|---|---:|
| log_H2O | −0.0585 |
| log_CO2 | **+0.0378** |
| log_CO | −0.0532 |
| log_CH4 | −0.0355 |
| log_NH3 | −0.0465 |

`mean|tanh(gate)| = 0.0463`. Bramka **wyrosła z dokładnego zera** przez 6 epok (~12,4 tys. kroków przy
`quantum_lr=2e-4`), a jej wartość końcowa jest niezerowa, czyli **oddaliła się** od zera. Kierunku zmiany
w czasie ten check nie ustala, bo trajektorii per epoka nie ma w artefaktach.

Czego brakuje, by to rozstrzygnąć: per-epokowe wartości bramki, run do konwergencji przy stałym
`quantum_scale=1.0` (bez rampy), ≥5 seedów.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a07_gate_dynamics.py` |
| komenda | `./.venv-qml/bin/python audit/a07_gate_dynamics.py` |
| kryterium PASS | bramka zbieżna (run zakończony własną regułą) ∧ znaki bramek per-gaz zgodne |
| wartość oczekiwana | `status = FAIL`; `mean_abs_tanh_gate = 0.046293`; `gate_raw = [−0.058534, +0.037774, −0.053296, −0.035546, −0.046500]`, `gate_tanh = [−0.058467, +0.037756, −0.053245, −0.035531, −0.046466]`; `zero_initialised = true`, `sign_agreement_across_gases = false` (CO₂ dodatni, cztery pozostałe ujemne), `converged_by_own_rule = false`; `epochs_run = 8`, `best_epoch = 6`, `max_epochs = 30`; `quantum_lr = 0.0002`, `approx_optimizer_steps_before_selection = 12426` (6 epok × ~2071 kroków, `weight_decay = 0.0001`); `circuit_weight_abs_mean = 0.091722` przy `circuit_init_scale = 0.1` |
| co by to obaliło | init inny niż `torch.zeros` (np. `randn`/`ones`) teza „wyrosła z zera" upada. `sign_agreement_across_gases = true` → `mean|tanh|` byłoby adekwatnym podsumowaniem i ta część zarzutu jest osłabiona. `converged_by_own_rule = true`, wtedy 0,046 jest stanem równowagi i część o „nie zdążył" upada. |
| rekord | `reports/audit/20260729/a07_gate_dynamics.json` |

Ustalenie ma **drugą ścieżkę odtworzenia**: one-liner poniżej czyta wyłącznie surowe artefakty
(`best_model.pt`, `config.json`) oraz źródło `models/ariel_exobiome/model.py`, bez `import audit_lib`, więc
weryfikuje samo ustalenie, a nie kod audytu.

```bash
cd ../hack4sages && grep -n "quantum_gate = nn.Parameter" -A 2 models/ariel_exobiome/model.py && ./.venv-qml/bin/python -c "
import torch, numpy as np, json
A='artifacts/ariel_quantum_best_v4_epoch6/'
sd=torch.load(A+'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
raw=sd['quantum_gate'].numpy(); g=np.tanh(raw); c=json.load(open(A+'config.json'))
for t,r,v in zip(['H2O','CO2','CO','CH4','NH3'],raw,g): print(f'{t:5}{r:+11.6f}{v:+11.6f}')
print(f'T1 mean|tanh| = {np.abs(g).mean():.6f}')
print(f'T2 zgodne znaki: {len(set(np.sign(g)))==1}')
print(f'T3 mnoznik |tanh|*0.5 w [{np.abs(g).min()*0.5:.4f}, {np.abs(g).max()*0.5:.4f}]')
print(f\"T4 ~{6*33138//c['batch_size']} krokow przy lr={c['quantum_lr']}\")"
```

| test | oczekiwane | co by obaliło K7 |
|---|---|---|
| grep | `nn.Parameter(torch.zeros(len(TARGET_COLUMNS)))` | `randn`/`ones`|
| T1 | 0,046293 | inna liczba |
| T2 | **False** — CO₂ `+`, cztery pozostałe `−` | `True` `mean\|tanh\|` byłoby adekwatnym podsumowaniem, zarzut osłabiony |
| T4 | ~12 426 kroków przy `lr = 2e-4` |  |

---

### K9. Niepełny wektor celu: brak temperatury i promienia.

**Gdzie znaleziony.** Głównym dowodem pomiarowym jest `audit/a15_target_completeness.py` — liczy
sprzężenia z `Tracedata.hdf5`, wyciek T przez aux na ADC, wartość kondycjonowania na prawdziwym
`(T, R_p)` oraz część cross-generator. Definicje celu i hardkody cech, na których te pomiary stoją,
są w kodzie modeli.

1. **Pięć vs siedem celów.** `models/ariel_exobiome/constants.py:29-35` — `TARGET_COLUMNS` zawiera
   dokładnie pięć pozycji (`log_H2O`, `log_CO2`, `log_CO`, `log_CH4`, `log_NH3`). Cel benchmarku ADC2023
   to **7 parametrów** (`planet_radius`, `planet_temp` + 5 gazów) — tyle ma `FM_Parameter_Table.csv`,
   `QuartilesTable.csv` i każdy trace w `Tracedata.hdf5`. Check `a15` porównuje oba wektory:

```python
TRACE_COLS = ["planet_radius", "planet_temp", *A.TARGETS]
...
"n_predicted": len(A.TARGETS), "n_required": len(TRACE_COLS)
```
**Hardkod w crossgen aux.**
2. `_build_taurex_auxiliary_frame` w `models/taurex_exobiome/dataset.py:315-335`
   wpisuje `star_temperature` i `planet_distance` jako stałe. Nie czyta `temperature_k` z etykiet
   (bliźniaczo w `taurex_exobiome_without_quant/dataset.py:327-347`):

```python
star_mass_kg = np.full(row_count, TAUREX_FIXED_STAR_MASS_KG, dtype=np.float32)
planet_distance = np.full(row_count, TAUREX_FIXED_PLANET_DISTANCE_AU, dtype=np.float32)
...
"star_temperature": np.full(row_count, TAUREX_FIXED_STAR_TEMPERATURE_K, dtype=np.float32),
...
"planet_distance": planet_distance,
```

3. **`temperature_k` nie jest cechą wejściową.** Oba pakiety crossgen biorą wyłącznie
   `frame[AUX_COLUMNS]` (`transform_aux_features` w `dataset.py:366-371`); lista cech
   (`constants.py:8-17`) zawiera `star_temperature` i `planet_distance`, ale **nie** `temperature_k`.
   To samo dotyczy FMPE: `AUX_FEATURE_COLS` w `taurex_fmpe/constants.py:24-33` nie zawiera
   `temperature_k`, a `raw_dataset.py:94-109` buduje `star_temperature` i `planet_distance` z hardkodu.
   `a15` przeszukuje cztery pakiety crossgen i potwierdza, że `temperature_k` nie jest w nich
   referencjonowane jako cecha wejściowa.

```python
AUX_COLUMNS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]
...
def transform_aux_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[AUX_COLUMNS].to_numpy(dtype=np.float32, copy=True)
```

4. **`ariel_winner_on_taurex`: `temperature_k` tylko w labelach.** `REQUIRED_LABEL_COLUMNS`
   (`prepare_dataset.py:40-48`) wymaga `temperature_k`, ale `build_auxiliary_matrix(...)` (`:113-130`)
   buduje aux z `FIXED_STAR_TEMPERATURE_K` i `FIXED_PLANET_DISTANCE_AU`:

```python
REQUIRED_LABEL_COLUMNS = (..., "temperature_k", ...)
...

def build_auxiliary_matrix(labels: pd.DataFrame) -> np.ndarray:
    row_count = len(labels)
    ...
    star_mass_kg = np.full(row_count, FIXED_STAR_MASS_KG, dtype=np.float32)
    planet_distance = np.full(row_count, FIXED_PLANET_DISTANCE_AU, dtype=np.float32)
    aux = np.column_stack(
        [
            np.full(row_count, FIXED_STAR_DISTANCE_PC, dtype=np.float32),
            star_mass_kg,
            labels["star_radius_rsun"].to_numpy(dtype=np.float32) * SOLAR_RADIUS_M,
            np.full(row_count, FIXED_STAR_TEMPERATURE_K, dtype=np.float32),
            ...
            planet_distance,
            ...
        ]
    )
```

**Po co w retrievalu w ogóle jest temperatura czyli trochę teorii.**

Line & Parmentier 2016 (arXiv:1511.09443, eq. 5) i HK17 (arXiv:1702.02051, eq. 12) rozdzielają dwie rzeczy:

| co widmo mierzy | od czego zależy w λ | rola `T` |
|---|---|---|
| **amplituda / „wysokość" cech** | wspólny mnożnik `H = kT/(μg)` na całe widmo | **główna** — `T` wchodzi liniowo w `H` |
| **kształt** (gdzie są pasma, stromość zboczy) | `d(ln σ_λ)/dλ` — fizyka linii; mieszanka `κ(λ) = Σ X_i σ_i(λ)` | **drugi rząd** — `σ(T)`, chmury, `μ` zależne od składu, profil T–P |

W eq. 5 abundancja **nie występuje** w pochodnej nachylenia — stąd sprzężenie T↔abundancja mierzone
w (a)–(b) idzie przede wszystkim przez **skalę wysokości i normalizację** (`P_ref·X`, `R_ref`), a nie
przez to, że `T` przesuwa linie w λ. Efekty drugiego rzędu mogą dodatkowo zmieniać kształt, ale to
nie jest główny mechanizm liczony w tabelach poniżej.

**ExoBiome nie przewiduje** `planet_temp` ani `planet_radius`. Na ADC `T` jest w praktyce
**wyciekana przez aux** (`r ≈ 0,99`, patrz (c)); na crossgen (tau) pozostaje **niekontrolowanym
parametrem** skalującym amplitudę wszystkich cech (patrz (d)). Pomiary (a)–(e) dotyczą **posteriora
ADC** (`Tracedata.hdf5`) oraz **generatora tau** w crossgen — **nie** `poseidon_test` (K1).

(a) Sprzężenie w posteriorze referencyjnym — ważone korelacje w trace'ach nested sampling, mediana po
663 planetach holdoutu — czyli na wszystkich, które maja 7-kolumnowa macierz trace:

| para | mediana \|r\| |
|---|---:|
| T ↔ abundancje (średnia po gazach) | **0.221** |
| R_p ↔ abundancje | 0.107 |
| gaz ↔ gaz | **0.053** |
| **R_p ↔ T** | **0.777** (mediana r = −0.777) |

Per gaz T sprzęga się najsilniej z CH₄ (0.445) i CO₂ (0.348), najsłabiej z H₂O (0.081), ale ogon
rozkładu sięga r = −0.774 (H₂O) i −0.855 (CO₂) — czyli dla istotnej mniejszości planet sprzężenie
jest bardzo silne. Dwa wnioski:

1. **T sprzęga się z abundancjami 4,14× silniej niż abundancje sprzęgają się między sobą.** Ma to
   bezpośrednią konsekwencję dla hierarchii odstępstw: rezygnacja z korelacji **gaz–gaz** (czyli
   z jointu na rzecz niezależnych marginalnych) dotyczy najsłabszej pary w macierzy, a pominięcie
   **T i R_p** — pary czterokrotnie silniejszej. Odstępstwem większym jest więc to drugie, mimo że
   pierwsze jest bardziej widoczne architektonicznie.
2. **R_p ↔ T to najsilniejsza korelacja w całej macierzy 7×7** — klasyczna degeneracja
   promień referencyjny ↔ temperatura — i **oba** te parametry są pominięte jednocześnie. Model nie
   ma więc żadnego uchwytu na dominującą degeneracją problemu.

(b) Kurs wymiany T → dex. Nachylenie kierunku degeneracji w posteriorze,
`d⟨log X⟩/dT = cov(T, log X)/var(T)`:

| gaz | mediana \|nachylenia\| [dex / 100 K] |
|---|---:|
| log_H2O | 0.215 |
| log_CO2 | 0.568 |
| log_CO | 0.591 |
| log_CH4 | 0.305 |
| log_NH3 | 0.262 |

Podane są mediany **modułu** nachylenia — tę wielkość emituje check
(`posterior_slope.per_gas_median_abs_slope`). Znak jest dominująco **ujemny**: mediana nachylenia ze
znakiem po wszystkich parach wynosi **−0.315 dex / 100 K**, czyli cieplejszy posterior = mniejsza
abundancja.

Mediana |nachylenia| po **wszystkich 3315 parach** gaz–planeta (663 planety, czyli cały dostępny materiał) = **0.352 dex / 100 K**, p90 = **1.000 dex / 100 K**. Liczba pochodzi z bloku `posterior_slope` w payloadzie `a15` i jest policzona na całym dostępnym materiale.
To jest bezpośredni pomiar **mechanizmu amplitudowego** z tabeli powyżej: obserwablą jest głębokość
cech ∝ `H = kT/(μg)`, więc błąd `T` mapuje się na błąd abundancji przez normalizację (`P_ref·X`,
`R_ref`), a nie przez zmianę położenia linii w λ. Podwojenie `μ` daje ten sam efekt na amplitudzie co
połowa `T` (Line & Parmentier 2016, eq. 5). Korekty kształtu (`σ(T)`, chmury) są tu drugiego rzędu —
nie tłumaczą głównej części nachylenia.

(c) Na ADC temperatura jest w praktyce WEJŚCIEM, nie parametrem ukrytym. Temperatura równowagowa
policzona z tabeli aux (`star_temperature`, `star_radius_m`, `planet_distance`) daje
`r = 0.9928`, RMSE **48.5 K** przy `sd(T) = 401 K`; GBM na samych 8 cechach aux → 62.8 K. Dlatego
podanie modelowi *prawdziwych* (T, R_p) poprawia gazy tylko o **0,3 %** (mRMSE 0.4354 → 0.4343) —
nie ma czego dodać, bo T już tam jest. Argument „brak T psuje dokładność" na ADC **nie broni się**.

I zostaje wniosek o samym benchmarku: **retrieval referencyjny nigdy nie widział tabeli aux.** Jego
własny posterior dla T ma medianę σ = 38.4 K przy RMSE mediany wobec prawdy **201.3 K** (przeufny
5,2×), podczas gdy aux daje T z błędem 48.5 K — **4× lepiej niż dokładny retrieval bayesowski**.
To kolejna, niezależna nitka wyjaśnienia K8: modele ML na ADC2023 dysponują informacją poboczną,
której referencja nie miała.

(d) Na zbiorze cross-generator temperatura jest nieobecna całkowicie — i to jest realny problem
fizyczny. `temperature_k` przebiega 500–1800 K (**3,6× w skali wysokości**), a
`_build_taurex_auxiliary_frame` hardkoduje `star_temperature = 5500 K` i `planet_distance = 0.05 AU`,
więc trasa przez temperaturę równowagową znika: `corr(T_eq_from_aux, temperature_k) = +0.0003`.
Jednocześnie `temperature_k` **nie jest czytane** przez `taurex_exobiome`,
`taurex_exobiome_without_quant` ani `taurex_fmpe` (w `ariel_winner_on_taurex` występuje wyłącznie na
liście `REQUIRED_LABEL_COLUMNS` — `models/ariel_winner_on_taurex/prepare_dataset.py:40` — też nie jest
cechą). Informacja jest przy tym w widmie: GBM na 218 binach przewiduje T z **R² = 0.872**
(RMSE 136 K). Czyli:

- model musi odtworzyć abundancje, marginalizując po nieznanym parametrze, który skaluje amplitudę
  wszystkich cech 3,6×,
- przy kursie z (b) 136 K błędu T to **0.479 dex** niejednoznaczności dla mediany gazu i **1.36 dex**
  w 90. percentylu — wobec raportowanego cross-generator RMSE 1.42 dex,
- nikt tego parametru nie przewiduje, nikt go nie podaje i nikt nie raportuje, pod jaką temperaturą
  abundancja została wyznaczona.

(e) Czego w tych danych NIE MA, a intuicja fizyczna tego oczekuje. W realnej atmosferze skład
jest powiązany z temperaturą przez chemię równowagową (przejście CO/CH₄, dysocjacja NH₃). W obu
generatorach **tego sprzężenia nie ma**: abundancje są losowane niezależnie od T
(`|corr(T, log X)| < 0.01` dla wszystkich pięciu gazów, w ADC i w crossgen). Sprzężenie widoczne w
posteriorze (a) i nachylenie z (b) pochodzą więc wyłącznie z geometrii tranzytu (`dα_λ/dλ ∝ H`,
Line & Parmentier 2016, eq. 5), nie z chemii równowagowej — jest **wyłącznie obserwacyjne** (skala
wysokości), nie chemiczne. To osobne
ostrzeżenie o realizmie zbiorów: model wytrenowany na tych danych nie może się nauczyć relacji
T ↔ skład, bo w danych jej nie ma — i nie jest za jej ignorowanie karany.

Dwa parametry, których brakuje, to dokładnie te, które Barstow i in. 2020 (arXiv:2002.01063)
retrievują obok tych samych pięciu gazów: promień na poziomie 10 bar i temperaturę „skali wysokości".
Ich zestaw parametrów to wasze pięć gazów plus dokładnie te dwa, które pomijacie — i to oni pokazują,
że informacja o temperaturze pochodzi z amplitud wszystkich cech molekularnych, więc nie da się jej
z problemu wyjąć.

Dlaczego to dyskwalifikuje: (i) niezgodność ze specyfikacją celu benchmarku — nie wolno pisać
„model dla ADC2023", przewidując 5 z 7 parametrów; (ii) abundancja bez towarzyszącej jej temperatury
**nie jest pomiarem, który da się zaraportować** — przy kursie 0.352 dex/100 K czytelnik nie potrafi
odtworzyć, pod jakim założeniem powstała liczba; (iii) na osi cross-generator jest to niekontrolowany
parametr zakłócający, a nie detal wyjściowy; (iv) darmowy sygnał nadzoru wielozadaniowego jest
wyrzucany — na ADC oba parametry odtwarzają się z R² 0.98–0.99, a na crossgenie temperatura z
**R² = 0.872** (promień znacznie słabiej, R² = 0.318).
**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a15_target_completeness.py` |
| komenda | `./.venv-qml/bin/python audit/a15_target_completeness.py` |
| kryterium PASS | model przewiduje pełny wektor parametrów benchmarku, albo pominięte parametry są jednocześnie słabo sprzężone **i** wyznaczone przez wejście |
| wartość oczekiwana | `n_predicted = 5`, `n_required = 7`, status FAIL, `status_terms.decisive = "coupling"`. Sprzężenie (**663 planety — cały dostępny materiał**): T↔gazy `0.22123`, R_p↔gazy `0.10748`, gaz↔gaz `0.05345`, iloraz T/gaz-gaz `4.1385`, R_p↔T `0.77660`; per gaz T: CH₄ `0.44515`, CO₂ `0.34799`, CO `0.11980`, NH₃ `0.11177`, H₂O `0.08142`, ogony p10 = −0.8553 (CO₂) i −0.7738 (H₂O). Kurs (blok `posterior_slope`, **3315 par**): mediana `0.352`, p90 `1.000` dex/100 K, iloraz do formy zamknietej `N = 7` → `1.058`. Wyciek T przez aux: `r = 0.99276`, RMSE `48.496 K`, `sd(T) = 401.385 K`. Kondycjonowanie: poprawa relatywna `0.00259`. Odtwarzalność ADC: R² `planet_radius = 0.99010` (RMSE 0.04539), `planet_temp = 0.97955` (RMSE 57.406 K). Crossgen: `corr(T_eq_from_aux, T) = +0.00031`, R² z widma `0.87218`, R² dla `planet_radius_rjup` `0.31807` |
| co by to obaliło |  gdyby `temperature_k` było podawane modelom na crossgenie jako cecha albo gdyby model miał 7 wyjść — upadają punkty (iii) i (iv)|
| rekord | `reports/audit/20260729/a15_target_completeness.json` |


### K9(f). Co przewidują modele w literaturze — i dlaczego 5 gazów nie ma precedensu

**Gdzie znaleziony.** To ustalenie jest o literaturze.

**Wyjaśnienie.** Przegląd 26 prac ML/SBI dla retrievalu atmosfer (2016–2026). Domyślny wektor celu w
tej literaturze to „temperatura + abundancje + kotwica rozmiaru/ciśnienia". Liczby celów grupują się
w dwóch reżimach: 5–10 parametrów dla transmisji (Márquez-Neila arXiv:1806.03944; Cobb
arXiv:1905.10659; ExoGAN arXiv:1806.02906; Yip arXiv:2011.11284; Ardévol Martínez arXiv:2203.01236;
Exoformer arXiv:2603.27623; ADC) i 16 parametrów dla emisji w petitRADTRANS (Vasist arXiv:2301.06575;
Gebhard arXiv:2312.08295), gdzie nadwyżkę stanowi mikrofizyka chmur i 4-punktowy
profil T–P.

Temperatura nie jest pomijana nigdy w głównym nurcie SBI. W 24 z 26 prac jest przewidywana.
Degeneracja ze skalą wysokości jest adresowana na jeden z trzech sposobów, nigdy przez milczące
pominięcie T: (a) dopasuj T, zamroź R₀/P₀ (Márquez-Neila, Cobb); (b) dopasuj T **i** P_ref
(Nixon & Madhusudhan arXiv:2004.10755 dla HD 209458b); (c) dopasuj T **i** R_p (ADC2023, ExoGAN, Yip,
Ardévol Martínez, Exoformer).

Dwa wyjątki — i tylko jeden jest uzasadniony.
- *Lueber i in. 2024* (arXiv:2405.02656) — random forest przewiduje 3 abundancje pierwiastkowe, bo siatka
  treningowa jest **samouzgodniona**: T nie jest parametrem swobodnym, jest funkcją etykiety.
  To legalny wzorzec dla retrievalu siatkowego. **U was nie zachodzi**: T jest losowana niezależnie.
- *INARA* (Soboczenski 2018, arXiv:1811.03390; Zorzan 2025) — 1D CNN z warstwą wyjściową `FC(12)`, same
  abundancje. Promień, masa, ciśnienie powierzchniowe i profil P–T **są losowane** przy generowaniu
  3 mln widm, ale nigdy nie są przewidywane, więc działają jako niemodelowana wariancja zakłócająca.
  Praca nie uzasadnia tego wyboru; autorzy sami nazywają komponent ML dowodem koncepcji.

Precedens na uzasadnione pominięcie istnieje, ale wygląda inaczej. Exoformer (arXiv:2603.27623) jako
jedyna praca jawnie uzasadnia usunięcie parametru (masa planety, z powodu degeneracji) — i **zamraża
go**, zamiast ignorować. To jest wzorzec, którego ExoBiome
mógłby użyć, ale to wymaga jawnej deklaracji i zamrożonej wartości.

Wniosek: zawężenie do 5 gazów jest w tej literaturze bez precedensu w klasie modeli, do której
ExoBiome należy (swobodna chemia, T losowana niezależnie). Dopuszczalne są dwie ścieżki: przewidywać
7 parametrów, albo jawnie kondycjonować na T i R_p i tak to nazwać.

**Co go weryfikuje.**

| | |
|---|---|
| check | ustalenie dotyczy literatury. |
| wartość oczekiwana | 26 prac w przeglądzie, **24 z 26 przewiduje T**; dwa wyjątki (arXiv:2405.02656 uzasadniony, arXiv:1811.03390 + Zorzan 2025 nieuzasadniony), następca INARA arXiv:2508.00076 dodaje 10 celów (6 abundancji + promień, grawitacja, ciśnienie, temperatura); dwa reżimy liczby celów: 5–10 (transmisja) i 16 (emisja pRT); INARA `FC(12)`, 3 mln widm (Zorzan: 3 112 620); Lueber 3 abundancje pierwiastkowe na siatce 1331 widm; Exoformer 6 celów.|
| co by to obaliło | znalezienie choćby **jednej** pracy z klasy ExoBiome (swobodna chemia, T losowana niezależnie od składu), która pomija T i to uzasadnia.|


### K9(g). Fizyka sprzężenia

**Gdzie znaleziony.** Ustalenie nie jest defektem w jednym pliku, jest wyprowadzeniem fizycznym plus
pomiarem. Pomiar: `audit/a15_target_completeness.py` na ważonych trace'ach z `Tracedata.hdf5`
(3315 par gaz–planeta). Konwencja ciśnienia referencyjnego, od której zależy `N`: domyślne
`atm_max_pressure` = 10 bar w TauREx3 `SimplePressureProfile`; w repo ta sama nastawa jest **jawnie
nadpisana** na 100 bar w generatorze cross-generator — `data/crossgen_biosignatures/constants.py:41`
(`PRESSURE_MAX_BAR = 1.0e2`), użyte w `data/crossgen_biosignatures/taurex_backend.py:126` i `:159-160`
(`atm_max_pressure=PRESSURE_MAX_BAR * 1.0e5`). Zdanie, które to ustalenie podważa, to samo milczenie o
temperaturze co w K9: `models/ariel_exobiome/constants.py:29-35` i `README.md:16`.

**Wyjaśnienie.** Co widmo transmisyjne w ogóle mierzy. Skala wysokości `H = k_B T / (μ m_u g)`, a
promień tranzytowy (Lecavelier des Etangs 2008; Heng & Kitzmann 2017, arXiv:1702.02051, eq. 12):

```
R(λ) = R_ref + H [ γ + E₁(τ_ref) + ln( P_ref κ(λ)/g · sqrt(2π R_ref/H) ) ],   γ = 0.5772
```

Amplituda cech (Line & Parmentier 2016, arXiv:1511.09443, eq. 5):

```
dα_λ/dλ = (2 R_p / R_*²) · H · d ln σ_λ / dλ
```

W tym przybliżeniu `T` i `μ` modulują **amplitudę** wszystkich cech przez wspólny czynnik `H`;
**kształt** widma w λ wynika z `d(ln σ)/dλ` (mieszanka absorberów `κ = Σ X_i σ_i`). Abundancja
**nie występuje** w eq. 5 — wchodzi wyłącznie przez `ln κ` w normalizacji offsetu (HK17), nie przez
nachylenie cech. Stąd trzy fakty, które wspólnie definiują problem:

1. **Wolno wnioskować tylko `P_ref · X`, nie `X`** (HK17: „it is `P₀χ`, rather than `χ`, that one
   really infers"), a `R_ref` wchodzi liniowo — więc mały błąd promienia daje ogromny błąd
   abundancji: 1 % w `R_p` → **2–4 dex** (Griffith 2014, arXiv:1312.3988); 3,4 % w `R₀` → **>5 dex**
   (HK17: `X_H₂O` 3,9·10⁻³ → 2,8·10⁻⁸).
2. **Z widma IR wolno wnioskować tylko abundancje *względne***. Benneke & Seager 2012
   (arXiv:1203.4018): stosunek `X_CH₄/X_CO₂` ograniczony do czynnika kilku przy 3σ, podczas gdy samo
   `X_CH₄` rozciąga się na **3 dex** (0,03 %–30 %). Absolutne wartości wymagają CIA i domknięcia
   `Σ X_i = 1` (Barstow i in. 2020, arXiv:2002.01063).
3. **`T`, `μ` i pokrycie chmurami są wzajemnie zdegenerowane przez amplitudę** (główny mechanizm).
   Line & Parmentier: podwojenie `μ` działa na `H` identycznie jak zmniejszenie `T` o połowę, a
   frakcja chmur wchodzi jako `(1−f)·H`. To samo miejsce w eq. 5. **Kształt** w λ rozdziela dopiero
   efekt drugiego rzędu: `σ(T)`, chmury spektralne, zmiana efektywnego `μ` gdy absorber dominuje
   (Novais i in. 2025, „banana" w posteriorze).

Wzmocnienie błędu temperatury. Z `R − R_ref = H·N`, `N = ln(P_ref/P_phot)`, przy zamrożonych
`R_ref`, `P_ref`:

```
Δ log₁₀ X = − (N / ln 10) · ε/(1+ε),      ε = ΔT/T
```

Dla konwencji `P_ref = 10 bar` (standard: Benneke & Seager `R_P,10`; Barstow i in. 2020; domyślne
`atm_max_pressure = 10 bar` w TauREx3 `SimplePressureProfile`) mamy `N ≈ 7`, czyli **±20 % w T daje
∓0,5 do ±0,75 dex**. Znak: **zimniej ⟺ mokrzej**.

WALIDACJA: pomiar w waszych danych zgadza się z formą zamkniętą. Kurs wymiany zmierzony z 3315
par gaz–planeta w `Tracedata.hdf5` (nachylenie kierunku degeneracji `cov(T, log X)/var(T)`), wobec
przewidywania analitycznego przy medianie posteriorowej `T = 887,3 K`:

| | dex / 100 K |
|---|---:|
| **zmierzone** (mediana po parach) | **0.352** |
| forma zamknięta, `N = 5` | 0.238 |
| **forma zamknięta, `N = 7`** (konwencja 10 bar) | **0.333** |
| forma zamknięta, `N = 9` | 0.428 |

Iloraz zmierzone/przewidziane przy `N = 7` wynosi **1.058**. Skalowanie `1/T` też się trzyma: zimna
połowa próbki **0.441**, gorąca **0.282** dex/100 K, iloraz **1.56** wobec przewidywanego **1.81**.

Zatem mechanizm jest zidentyfikowany jednoznacznie: **skala wysokości i degeneracja normalizacyjna**,
nie chemia. Wielkość efektu jest potwierdzona dwiema niezależnymi drogami.

Kotwice literaturowe na wielkość skażenia. Rocchetto i in. 2016 (arXiv:1610.02848): założenie
izotermy zawyża abundancje o **~1 rząd** i zaniża słupki błędu → wartości **6–11σ** od prawdy.
MacDonald, Goyal & Lewis 2020 (arXiv:2003.11548): dopasowanie modelu 1D do asymetrycznego terminatora
daje T niższe o **~1000 K** i `log X_H₂O` **~2 dex** za wysokie (skład słoneczny mylony z 15× powyżej
słonecznego przy >3σ). Welbanks & Madhusudhan 2019 (arXiv:1904.05356), HD 209458 b, przypadek 1→2:
**ΔT = −933 K towarzyszy Δlog X = +4,25 dex**.

Kluczowa granica licencji na upraszczanie. Welbanks & Madhusudhan pokazują, że degeneracja
`R_p`–`P_ref` jest **jednowymiarowa** o nachyleniu `d log₁₀P_ref/dR_p = −1/(H ln 10)` i wolno
zamrozić jedną z tych dwóch wielkości. Ale to nachylenie **jest wyznaczone przez `H`, czyli przez
`T`**. Jeśli `T` jest nieznana, nachylenie jest nieznane i degeneracja 1-D rozdyma się z powrotem do
2-D. **Licencja dotyczy `R_ref` vs `P_ref`, nie temperatury.** W całym przeglądzie nie ma pracy, która
uznawałaby wnioskowanie abundancji bez stopnia swobody na temperaturę za obronione; każda, która to
zbadała, raportuje skażenie na poziomie 1σ–3σ.

Struktura rozkładu, której punkt nie może wyrazić. Novais i in. 2025 (arXiv:2503.04600) opisują
łączny posterior `(T, log X_H₂O)` jako **„banana-shaped"**: antyskorelowany dla `log X ≲ −2` (tam
tylko T zmienia skalę wysokości) i **dodatnio** skorelowany powyżej (absorber zaczyna dominować `μ`,
co tłumi cechy i wymusza wzrost T). Estymator punktowy musi uśrednić po obu gałęziach. Line &
Parmentier znajdują wręcz **dwumodalny** posterior `X_H₂O` (mod przyokołosłoneczny z ~50 % chmur vs
mod wysoko-`μ` bez chmur), odległe o ≳2 dex i statystycznie nierozróżnialne (`ln B = 0.7`). Cubillos
& Blecic 2021 (arXiv:2105.05598) potwierdzają to populacyjnie na ~50 widmach WFC3: silne korelacje
dają szerokie, wielomodalne lub nieograniczone marginalne, a T koreluje silnie z wysokością
referencyjną na 0,1 bar.

Najbliższy opublikowany analog ExoBiome dokumentuje dokładnie tę awarię. Yip i in. 2021
(arXiv:2011.11284), deterministyczne DNN na widmach Ariel Tier-2 z 9 celami: `T_p` **systematycznie
zaniżane powyżej 1500 K**, `R_p` systematycznie zaniżane, `log P_cloud` zapada się do dwóch poziomów.
Ich diagnoza to wprost nieadekwatność estymatora punktowego wobec kompensujących się parametrów; ich
mapy czułości pokazują, że sieć **odczytuje `H` z rozmiaru cech** i tak wyprowadza temperaturę.

I samokrytyka zwycięzcy, wprost o tym sprzężeniu. Aubin i in. 2023 (arXiv:2309.09337) raportują, że
referencyjny posterior nested sampling ma człon diagonalny między temperaturą a udziałem masowym H₂O,
którego ich zwycięski model nie uczy się — a utrata tej informacji nie kosztuje nic pod metryką KS.
Dodają, że nie oczekują, by zwycięski model wydobył spójne parametry atmosferyczne z realnych widm.

Co więc ExoBiome zrobił z temperaturą. Nie zmarginalizował — **skondycjonował na priorze
treningowym i zapadł do punktu**. Regresor na L2 zbiega do `E_π[log X | d]`, gdzie `π` to rozkład
próbkowania syntetycznego zbioru. Konsekwencje: (i) tam, gdzie widmo nie ogranicza `log X` przy
nieznanym `T`, wyjście zwraca średnią prioru na rozmaitości degeneracji, więc **podłoga i sufit
abundancji są własnością generatora, nie widma** — Fisher & Heng 2018 (arXiv:1809.06894) pokazują
dokładnie ten mechanizm w retrievalu bayesowskim, gdzie dolne ograniczenia na `X` są artefaktami
prioru na `R₀`/`P₀`, a degeneracja `X_H₂O` rzędu wielkości wynika ze zmian `R₀` w trzeciej cyfrze
znaczącej; (ii) punkt nie potrafi reprezentować banana ani dwumodalności, więc może wypaść w obszarze,
który dane aktywnie odrzucają; (iii) **wynik jest niefalsyfikowalny** — bez `T` i `R_ref` nie ma czego
podstawić do forward modelu, więc nie ma widma resztkowego, χ², ani evidence. Nie da się odróżnić
„widmo ogranicza H₂O do −4,9 ± 0,3" od „widmo jest linią prostą i model zwrócił średnią prioru".

Uwaga o wkładzie własnym. Przegląd nie znalazł **żadnej** pracy podającej liczbowy współczynnik
korelacji `T` ↔ `log X` dla posteriorów transmisyjnych — literatura jest w tym punkcie wyłącznie
jakościowa („strong", „banana-shaped", „inversely correlated") plus corner ploty. Pomiary z `a15`
(med |r| = 0.221 ogółem, 0.081–0.445 per gaz; kurs 0.352 dex/100 K) są więc
**liczbą, której w literaturze nie ma**, policzoną z publicznie dostępnego zbioru referencyjnego.
To jest samodzielny, cytowalny wkład — i najmocniejszy element narracji metodologicznej z §7.2.


**Co go weryfikuje.**

| | |
|---|---|
| check | **brak checku dla samego wyprowadzenia** — wzór nie jest własnością repo. Pomiar, z którym się go zestawia, pochodzi z `audit/a15_target_completeness.py`. |
| komenda | `./.venv-qml/bin/python audit/a15_target_completeness.py` (część pomiarowa); dla wyprowadzenia — `—` |
| kryterium PASS | (dla części pomiarowej, z `a15`) model przewiduje pełny wektor parametrów benchmarku, albo pominięte parametry są jednocześnie słabo sprzężone **i** wyznaczone przez wejście |
| wartość oczekiwana | kurs zmierzony **0.352 dex/100 K** na **3315 parach** gaz-planeta (663 planety) przy medianie posteriorowej `T = 887,3 K`; forma zamknięta `N = 5` → 0.238, `N = 7` → **0.333**, `N = 9` → 0.428; iloraz zmierzone/`N = 7` = **1.058**; skalowanie `1/T`: zimna połowa `0.4406` vs gorąca `0.2824`, iloraz **1.56** wobec przewidywanego **1.81**; wartości podpierające z `a15`: med \|r\| T↔gazy `0.22123`, per gaz `0.08142`–`0.44515`. Sam wzór odtwarza `>5 dex` HK17 (daje 5,15) i dolny koniec zakresu Griffith (2,13 dex vs jej „~100") |
| co by to obaliło | (1) **to jest WYPROWADZENIE, nie cytat** — wzór `Δlog X = −(N/ln10)·ε/(1+ε)` postawił asystent i w tej formie **nie występuje w żadnej z przeczytanych prac** (ograniczenie zapisane w notatce MP: wzor jest wyprowadzeniem, nie cytatem). Zgodność **0,352 vs 0,333** dex/100 K (iloraz **1,058**) jest **potwierdzeniem, nie dowodem**, i poprawa zgodnosci z 1,10 na 1,058 po zdjeciu capa **podnosi stawke na V7, nie obniza** — bo forma zamknieta i pomiar **dziela zalozenie**, ze degeneracja idzie przez skale wysokosci przy `P_ref` = 10 bar, więc lepsza zgodnosc jest dowodem na `N ≈ 7`, a nie niezależnym potwierdzeniem wzoru: błąd algebraiczny w wyprowadzeniu przy jednoczesnym trafieniu w rząd wielkości jest możliwy, bo `N` jest parametrem swobodnym dopasowanym konwencją. Rachunek trzeba przejść samodzielnie przed cytowaniem; (2) **V7 jest nierozstrzygnięte**: czy TauREx3 definiuje `planet_radius` na poziomie `atm_max_pressure` = 10 bar (do sprawdzenia w źródle, nie w dokumentacji — zadanie A1.6). To ustala `N ≈ 7` i **cały kurs**. Inna definicja poziomu referencyjnego → inne `N` → inny kurs: przy `N = 5` przewidywanie spada do 0.238 (iloraz 1,51), przy `N = 9` rośnie do 0.428 (iloraz 0,84). Konkretne ostrzeżenie z repo: własny generator projektu **nie** używa 10 bar, lecz `PRESSURE_MAX_BAR = 1.0e2` (`data/crossgen_biosignatures/constants.py:41`), czyli 100 bar — dowód, że domyślna konwencja nie jest w tym projekcie uniwersalna; (3) gdyby `N` implikowane wypadło poza fizycznie sensownym zakresem, walidacja przestaje być walidacją — dziś p25 = 4.53 i p75 = 13.09 obejmują 7, ale rozrzut jest szeroki; (4) gdyby posterior referencyjny okazał się niereprezentatywny (podzbiór 663 z 4143, 16 % — A0.6), zmierzone 0.352 dotyczyłoby planet nietypowych; (5) twierdzenie o wkładzie własnym („żadna praca nie podaje liczbowego `corr(T, log X)`") jest **twierdzeniem negatywnym niezweryfikowanym** (V6/A1.4) — jedna praca z taką liczbą je obala |
| rekord | `reports/audit/20260729/a15_target_completeness.json` (część pomiarowa); wyprowadzenie — brak rekordu maszynowego, `docs/LITERATURA.md` Tab. 3 + nota pod nią |

---

### K10. Raportowana metryka nie jest metryką benchmarku

**Gdzie znaleziony.** Oficjalny kod punktujący leży w repo: `models/adc_baseline/posterior_utils.py:84`
(`score_trace.append((1 - metric_ks.statistic) * 1000)`) oraz `models/adc_baseline/spectral_metric.py:19`
i `:43` (`score = 1000-np.mean([bound_loss,median_loss])`). Liczby raportowane jako wynik to mRMSE z
`reports/model_comparison/rmse/exobiome_metrics.json`, `sota_metrics.json` i `cnn_metrics.json`.

**Wyjaśnienie.** Oficjalna punktacja ADC2023 to

> `score = 0.8 × score_posterior + 0.2 × score_spectral`

gdzie `score_posterior` to **dwupróbkowy test Kołmogorowa–Smirnowa na 7 rozkładach marginalnych**
wobec próbek referencyjnych MultiNest, a `score_spectral` to odwrócona strata Hubera na medianie
widma i IQR. Potwierdzone niezależnie przez trzy prace uczestników (Aubin arXiv:2309.09337; Unlu i in.
arXiv:2310.10521; Sweet arXiv:2406.10771). **mRMSE nie jest metryką tego challenge'u w żadnym stopniu.**

Konsekwencje, każda osobno poważna:

1. **„ExoBiome bije zwycięzcę ADC2023" jest zdaniem o metryce, której challenge nie używał**, i której
   zwycięzca nie optymalizował. Zestawione z K3 (różne wejścia) i K8 (odwracanie symulatora) to
   **trzeci niezależny powód**, dla którego to porównanie jest nieważne.
2. **Model punktowy jest na tej metryce niepunktowalny.** KS wymaga rozkładu; ExoBiome go nie
   produkuje. Nie da się więc podać liczby, która stawiałaby ExoBiome na tej samej skali co
   zgłoszenia challenge'owe .
3. **Metryka wymaga 7 marginalnych** — przy 5 wyjściach nie da się jej policzyć z definicji (K9).
4. Atut dla waszej narracji metodologicznej: Aubin i in. sami stwierdzają, że metryka mierzy tylko
   dopasowanie marginalnych, nie rozkładu łącznego, i że ich model o wyższym potencjale wypadł w
   rankingu **niżej**. Wprost wzywają do ponownej oceny metryki. To jest najmocniejszy istniejący
   opublikowany zarzut wobec bodźców tego benchmarku.

Dodatkowo, do sekcji o luce sim-to-real: ukryty zbiór testowy ADC2023 zawierał chmury, dodatkowy
gatunek chemiczny, aktywność gwiazdową i niejednorodne profile T — czyli misspecyfikację, której
7-parametrowa przestrzeń celu nie potrafi wyrazić. Żaden model challenge'owy nie był testowany na
realnych widmach.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a24_official_metrics.py` (część pomiarowa). Sama **specyfikacja metryki** nie ma i nie może mieć checku — potwierdzają ją trzy prace uczestników (arXiv:2309.09337, arXiv:2310.10521, arXiv:2406.10771) i oficjalny kod w repo (`models/adc_baseline/posterior_utils.py:84`, `models/adc_baseline/spectral_metric.py:19,43`);|
| komenda | `./.venv-qml/bin/python audit/a24_official_metrics.py` |
| kryterium PASS | model punktowy ma **dodatnią skill wobec ramienia PRIOR** na KS, czyli bije rozkład ignorujący widmo. Poprzednie kryterium („wynik w 2× podłogi skończonej próbki") zostało **wycofane jako niefalsyfikowalne**: delta Diraca ma KS ≥ 0,5 z konstrukcji, a podłoga wynosi ~0,044, więc warunku nie mogła spełnić żadna para gaz–planeta — payload potwierdza `0 z 3315`. Nowe kryterium działa w obu kierunkach w jednym przebiegu: NSF je przechodzi (**+0,276**), ExoBiome nie (**−0,231**) |
| wartość oczekiwana | status FAIL. Wzór: `0.8 × score_posterior + 0.2 × score_spectral`, `score_posterior` = KS na **7** marginalnych wobec próbek MultiNest, `score_spectral` = odwrócona strata Hubera na medianie widma i IQR, każdy skalowany 0–1000. Model przewiduje 5 wyjść, więc metryki nie da się policzyć z definicji; `not_scored` w payloadzie `a24` = `planet_radius`, `planet_temp`, `ADC2023 spectral component`. Wejście challenge'u: 52 biny IR + 9 cech pomocniczych |
| co by to obaliło | (1) oficjalny dokument organizatorów, który podawałby mRMSE jako metrykę ADC2023, albo **6** celów zamiast 7 — strona challenge'u w `/documentation/data` podaje wciąż 6 (nieaktualna treść ADC2022) i **nie ma oficjalnego papieru ADC2023 na arXiv ani w czasopiśmie**; rozstrzyga to V8/A1.5. Dopóki to nie jest domknięte, specyfikacja stoi na trzech pracach uczestników plus na kodzie zawendorowanym w `models/adc_baseline/`, nie na dokumencie organizatorów; (2) przeformułowanie claimu z „bijemy zwycięzcę ADC2023" na „punktowa dokładność na poziomie SOTA przy N× niższym koszcie" (wariant, który `docs/VERIFICATION.md:128-130` sam dopuszcza) — wtedy zdanie przestaje być zdaniem o metryce challenge'u i punkt 1 przestaje mieć adresata; (3) punkt 2 upada dopiero po dodaniu głowy rozkładowej **i** siedmiu wyjść jednocześnie — samo jedno z dwóch nie wystarcza |
| rekord | `reports/audit/20260729/a24_official_metrics.json` |


### K10(b). Oficjalny baseline challenge'u jest w repo — i produkuje rozkład

**Gdzie znaleziony.** `models/adc_baseline/` — cały katalog; głowa rozkładowa w
`models/adc_baseline/MCDropout.py:17` oraz `:33` i `:35` (`Dropout(p)(x,training=True)`). Kod
punktujący: `models/adc_baseline/posterior_utils.py`, `models/adc_baseline/spectral_metric.py`. Liczba,
której status to koryguje: `models/adc_baseline/cnn_metrics.json` (`rmse_mean = 0.65003745144915`),
identyczna po sparsowaniu z `reports/model_comparison/rmse/cnn_metrics.json`.

**Wyjaśnienie.** `models/adc_baseline/` to zawendorowana kopia `ucl-exoplanets/ADC2023-baseline`:
`MCDropout.py`, `posterior_utils.py`, `spectral_metric.py`, `FM_utils_final.py`, `run_baseline.py`,
`submit_format.py`, `preprocessing.py`, `helper.py`. Lista plików
zgadza się z repozytorium organizatorów.

Architektura: CNN zmodyfikowana z Yip i in. (arXiv:2011.11284), z **Monte Carlo dropout na etapie
testu** — `Dropout(p)(x, training=True)`, czyli dropout aktywny w inferencji — co daje **rozkład
wielowymiarowy dla każdego przykładu testowego**. Oficjalny baseline jest więc modelem
**rozkładowym**.

Trzy konsekwencje, wszystkie istotne:

1. **Baseline organizatorów istnieje i jest punktowalny na metryce challenge'u** w odróżnieniu od
   ExoBiome.
2. **Liczba 0,650 nie jest bez pokrycia.** `models/adc_baseline/cnn_metrics.json` jest identyczny
   po sparsowaniu z `reports/model_comparison/rmse/cnn_metrics.json`, a katalog zawiera kod, wagi
   i notebook.
3. **MC-dropout to tania i legalna głowa niepewności.** Memo o architekturze punktowej traktuje
   dodanie niepewności jako pozycję odłożoną. Organizatorzy zrobili to jednym argumentem
   `training=True` w warstwie dropout, na CNN o 2,3 MB wag.

**Co go weryfikuje.**

| | |
|---|---|
| check | **brak checku** — weryfikacja przez `models/adc_baseline/`: `MCDropout.py:17,33,35` (dropout aktywny w inferencji), `posterior_utils.py`, `spectral_metric.py`, oraz porównanie listy plików z `ucl-exoplanets/ADC2023-baseline`. |
| komenda | `—` |
| kryterium PASS | '-' |
| wartość oczekiwana | `cnn_metrics.json`: `rmse_mean = 0.65003745144915`, per gaz H₂O `0.80920`, CO₂ `0.53405`, CO `0.60904`, CH₄ `0.49332`, NH₃ `0.80457`. W `a13`: `cnn_holdout` → `status: backed_summary`, `artifacts_present = {metrics: true, predictions: false, weights: true}`, wagi `models/adc_baseline/cnn_whole_ariel_new.weights.h5`. |
| co by to obaliło | (1) gdyby `training=True` nie było przekazywane w inferencji, baseline nie byłby rozkładowy i punkty 1 oraz 3 upadają — to jedna linia do sprawdzenia (`models/adc_baseline/MCDropout.py:17`); (2) gdyby zawendorowana kopia różniła się od `ucl-exoplanets/ADC2023-baseline`|
| rekord | `—` |

---

### K11. TauREx też ma wadę fizyczną: brak absorpcji indukowanej zderzeniami (CIA)

**Gdzie znaleziony.** `data/crossgen_biosignatures/taurex_backend.py:154-157` — lista `contributions`
przekazana do `TransmissionModel` ma dwa elementy i nie ma wśród nich `CIAContribution`. Ten sam
generator, ta sama wada, drugie wywołanie: `data/crossgen_biosignatures/taurex_backend.py:120-126`.
Zdania, które to podważa: wszystkie liczby cross-generator raportowane jako **bezwzględne** abundancje
oraz nazwa „cross-generator gap" w `reports/` — gap ma mierzyć różnicę implementacji transferu
promienistego.

**Wyjaśnienie.** `data/crossgen_biosignatures/taurex_backend.py:154-157`:

```python
contributions=[
    AbsorptionContribution(),
    RayleighContribution(),
]
```

Brakuje `CIAContribution`. W atmosferze zdominowanej przez H₂/He absorpcja indukowana zderzeniami
H₂–H₂ i H₂–He jest **głównym źródłem kontinuum w bliskiej podczerwieni**.

Konsekwencje:
1. **Cały zbiór tau (41 423 widma, czyli trening i walidacja) jest generowany bez głównego źródła
   kontinuum.** Model uczy się odwracać forward model, który nie odpowiada żadnej normalnej
   konfiguracji retrievalowej — również tego samego TauREx-a. Referencją dla tego, jak konfiguracja
   TauREx-a powinna wyglądać (CIA, ciśnienie referencyjne, listy linii), jest Waldmann i in. 2014
   (arXiv:1409.2312).
2. **Porównanie międzygeneratorowe traci nawet swoją nazwę.** Jeśli TauREx liczy bez CIA, a poprawnie
   wywołany POSEIDON liczyłby z CIA, to „gap" nie mierzy różnicy implementacji transferu
   promienistego, tylko brak jednego członu opacity po jednej stronie.

POSEIDON w tej konfiguracji włącza CIA domyślnie — asymetria jest
jednostronna. `poseidon_backend.py:103-114` woła `define_model(...)` bez żadnego przełącznika CIA
(`bulk_species=["H2","He"]`, ślady = 5 gazów, brak argumentu `disable_continuum`). W `define_model`
CIA **nie jest opcją wywołania** — biblioteka buduje `CIA_pairs` sama, filtrując `supported_cia` po
tym, które gazy są obecne w `chemical_species = bulk_species + param_species`:

```
# POSEIDON/core.py:672-678 (zainstalowany pakiet, /private/tmp/poseidon-venv, poseidon-1.4.0.dist-info)
CIA_pairs = []
for pair in supported_cia:
    pair_1, pair_2 = pair.split('-')
    if (pair_1 in chemical_species) and (pair_2 in chemical_species):
        CIA_pairs.append(pair)
```

Dla `chemical_species = [H2, He, H2O, CO2, CO, CH4, NH3]` i `supported_cia` z
`POSEIDON/supported_chemicals.py:34-36` włączają się **6 par**: `H2-H2`, `H2-He` (dominujące kontinuum tła), `H2-CH4`,
`CO2-H2`, `CO2-CO2`, `CO2-CH4`. `disable_continuum` domyślnie `False` (`core.py:1305`), więc nic
tego nie wyłącza; `extinction(...)` dodaje CIA do κ zawsze, gdy `CIA_pairs` niepuste.

NOTE: w odróżnieniu od K1 dane mają strukturę i model się na nich
uczy (skill 0,50 na tau/val). Ale każde twierdzenie o **bezwzględnych** abundancjach z tego zbioru
dziedziczy przesunięcie rzędu dex.

Dodatkowe dowody na kodzie **spoza** audytu.

*Oryginalny kod organizatorów challenge'u, w repo.*
`models/adc_baseline/FM_utils_final.py:165-211` to **oficjalny forward model ADC2023** (docstring:
„Initialise the official forward model for ADC2023"), czyli ta sama biblioteka i to samo pasmo, na
których powstał zbiór ADC. Konfiguruje **trzy** wkłady i osobno ładuje bazę CIA:

```python
# FM_utils_final.py:181, 207-209
CIACache().set_cia_path(CIA_path)
...
forward_model.add_contribution(AbsorptionContribution())
forward_model.add_contribution(RayleighContribution())
forward_model.add_contribution(CIAContribution())
```

*Kotwica B — źródło TauREx3, wywołanie odtworzone.* W zainstalowanym `taurex 3.3.2`
(`.venv-cnn`) lista wkładów startuje **pusta** (`taurex/model/model.py:42`:
`self.contribution_list = []`) i jedyną drogą wejścia jest `add_contribution`
(`model/model.py:57-65`), wołane albo jawnie, albo z pętli po `contributions=`
(`model/simplemodel.py:121-123`). Żadna gałąź nie dokłada wkładu implicite. Odtworzone oba wywołania:

```
NASZE      (taurex_backend.py:154-157)   → ['Absorption', 'Rayleigh']
OFICJALNE  (FM_utils_final.py:207-209)   → ['Absorption', 'Rayleigh', 'CIA']
```

Czyli brak `CIAContribution` w liście **oznacza brak CIA w widmie**.

**Co go weryfikuje.**

| | |
|---|---|
| check | **brak checku**, weryfikacja przez **lekturę trzech źródeł spoza audytu**: `data/crossgen_biosignatures/taurex_backend.py:154-157` (nasze wywołanie), `models/adc_baseline/FM_utils_final.py:181,207-209` (oficjalny FM ADC2023 organizatorów — trzy wkłady + `CIACache`), oraz `taurex/model/model.py:42,57-65` + `simplemodel.py:121-123` w zainstalowanym `taurex 3.3.2` (brak domyślnego CIA), z odtworzeniem obu wywołań |
| wartość oczekiwana | Z lektury kodu: `contributions` ma **2** elementy (`AbsorptionContribution`, `RayleighContribution`), zero wystąpień `CIAContribution` w `data/crossgen_biosignatures/`; zbiór tau **41 423** widma (`tau/train 37 281` + `tau/val 4 142`), pasmo 0,6–5,2 µm, 218 binów, `PRESSURE_LEVELS = 100`, `PRESSURE_MIN_BAR = 1.0e-6`, `PRESSURE_MAX_BAR = 1.0e2`; tło H₂/He: `vmr_h2 ∈ [0.8275, 0.8500]`, `vmr_he ∈ [0.1460, 0.1500]` (`a21`).|
| co by to obaliło | Wszystkie z tych wątpliwości zostały zamknięte. |
| rekord | `—` |

---

## 7. Werdykt i plan ratunkowy

### 7.1 Czego nie da się uratować

Obecnej narracji QML; obecnej osi cross-generator; obecnej tabeli benchmarku SOTA; obecnego flagowego
checkpointu jako wyniku; rankingu z `taurex_model_comparison.md`; wykresu czterech modeli; benchmarku
low-resource vs FMPE.

### 7.2 Co jest realną publikacją

1.**Cross-generator gap jako misspecyfikacja modelu** (Ward i in., arXiv:2210.06564) — zmierzony *po*
   naprawie danych, w sparowanym designie, ze skill score, przy zrównanych forward-modelach
   (TauREx: Waldmann i in., arXiv:1409.2312).
2.Opis generatorów, zebranie ich w jedną publikację, przeanalizowanie modelu szumu względem realnych danych z JWST.


### 7.3 Minimalna lista poprawek

Kolejność jest istotna — punkty 1–4 są warunkami wstępnymi.

1. **Zregenerować POSEIDON**, dodać do walidatora asercję zmienności spektralnej i obowiązkowy ridge smoke
   test przed każdym commitem datasetu, oraz hash treści danych w kluczu cache'u, weryfikacja `a01`
2. **Dodać do każdej tabeli predyktor stały i skill score.** Nie raportować rankingów, w których jakikolwiek
   model ma skill ≤ 0, weryfikacja `a02`
3. **Zamrozić jedną konwencję wejścia** i zaraportować wszystkie modele w obu. Przetrenować ExoBiome z
   augmentacją szumu, bo bez tego nie ma porównania z żadną metodą SBI, weryfikacja `a03`
4. **Naprawić final eval `quantum_scale`**, przeliczyć każdą liczbę kwantową, weryfikacja `a04`, `a13`
5. **≥5 seedów dla każdego porównania**, wariancja po treningach a nie po próbkach, korekta na wielokrotność
   zadeklarowana z góry, weryfikacja `a12`
6. **Doprowadzić baseline do konwergencji na metryce porównania**: early stopping na mRMSE, pełna walidacja,
   `point_estimate=mean`, udokumentowany budżet compute (łącznie z brakującym stage-1), weryfikacja `a05`
???7. **Naprawić `taurex_fmpe` holdout ≡ validation** i wydzielić prawdziwy test in-domain na TauREx.
   → weryfikacja `a10`
8. **Jeśli wracać do kwantu**: jedno porównanie, param-matched, symetryczny budżet. Inna konstrukcja gałęzi kwatnowej
   większy wkład kwantu, weryfikacja `a06`
9. **Głowa niepewności do rdzenia** — ale dopiero po 1–3; bez zamrożonej konwencji szumu kalibracja nie ma
    względem czego być skalibrowana.

---

## 8. Inwentarz plików potrzebnych do udowodnienia i naprawienia zarzutów
(czysty przebieg `20260729`: 21 checków, `FAIL 17 / WARN 1 / INFO 1 / PASS 2`, 0 ERROR — `summary.json`)

```
audit/audit_lib.py                   provenance (git sha, env, sha256), metryki, skill score,
                                     rekonstrukcja pipeline'u ExoBiome, paired bootstrap, MDE
audit/run_all.py                     uruchamia suite, składa reports/audit/<data>/summary.{json,md},
                                     exit != 0 przy jakimkolwiek FAIL -> brama pre-commit
audit/README.md                      spec wszystkich checków + statusy + skrypty naprawcze
audit/a01_spectral_variation.py      K1
audit/a02_trivial_baseline.py        K2
audit/a03_input_convention.py        K3   (tabela 2x2 szum x model)
audit/a04_quantum_scale_provenance.py K4  (sweep skali + ablacja gate-off w dex)
audit/a05_training_completeness.py   K5 + niedotrenowanie baseline'u
audit/a06_param_accounting.py        K6
audit/a07_gate_dynamics.py           K7
audit/a08_reference_posterior.py     K8 + sonda multimodalności z Tracedata.hdf5
audit/a09_noise_realization.py       W14 (statystyka R z kalibracją na noiseless/noisy)
audit/a10_split_integrity.py         P1, P2, U4
audit/a11_pairing_audit.py           P3
audit/a12_significance_power.py      P7, P11
audit/a13_provenance_index.py        P6, U8, U9
audit/a14_importability.py           U6
audit/a15_target_completeness.py     K9   (sprzężenie T-abundancje, wyciek T przez aux)
audit/a21_dead_features.py           U2, U3, K9(d)
audit/a24_official_metrics.py        K10  (metryki ADC2022/ADC2023)
audit/a26_baseline_ladder.py         K2 rozszerzone — drabina stała/aux/widmo/oba
audit/a29_smoke_baseline_recovery.py K1(b) — dopisane PO przebiegu 2026-07-27
```

Poza `run_all.SUITE`, każdy z własnego powodu:

```
audit/a27_pipeline_fidelity.py       P8(b) — wierność rekonstrukcji pipeline'u (zadanie A0.2)
audit/d01_poseidon_diagnosis.py      K1(c) — POSEIDON-a nie ma ani w .venv-qml, ani w .venv-cnn,
                                     więc w suite byłby błędem przy każdym przebiegu; osobny venv,
                                     komenda w audit/README.md
```


## Załącznik A. Liczby kluczowe w jednym miejscu

| wielkość | wartość | źródło |
|---|---:|---|
| ExoBiome params / winner params | 258 688 / 10 771 200 (41,6×) | `a06`, `a03` |
| obwód / model | 24 / 258 688 = 0,0093 % | `a06` |
| ścieżka kwantowa vs classical-only | +69 410 par. = +36,7 % | `a06` |
| `mean|tanh(gate)|` | 0.0463 (CO₂ przeciwny znak) | `a07` |
| ExoBiome holdout, kwant OFF | 0.302409 | `a04` |
| ExoBiome holdout, skala 0.5 (walidowana) | 0.295552 | `a04` |
| ExoBiome holdout, skala 0.6667 (optimum sweepu) | 0.295487 | `a04` |
| ExoBiome holdout, skala 1.0 (raportowana) | 0.298693 | `a04` |
| wkład ścieżki kwantowej: @1.0 / @0.5 | 0.003716 / 0.006857 | `a04` |
| ExoBiome holdout, wejście +N(0,σ) | 0.966512 ± 0.006572 | `a03` |
| NSF holdout, wejście czyste, mean / median | 0.403603 / 0.384621 | `a03` |
| NSF holdout, wejście +N(0,σ), mean / median | 0.548927 / 0.556800 | `a03` |
| stosunek NSF/ExoBiome, ramię **median** (opublikowane): czyste / zaszumione | 1.288 / 0.576 (odwrócenie) | `a03` |
| baseline stały: ADC holdout / TauREx val / POSEIDON | 1.4404 / 2.8852 / 2.8940 | `a02` |
| skill: quantum / noquant / winner / H200 na POSEIDON | −0.111 / −0.133 / −0.193 / −0.000 | `a02` |
| udział **tabeli** aux w skillu (ADC / tau-val) | **−0,19 % / −0,41 %** — tabela nie dodaje skillu **ponad to, co widmo już koduje** (patrz K2(b): `star_temperature` odtwarza się z wektora szumu, R² = 0,749) | `a26` |
| GBM na widmie vs ExoBiome (ADC holdout) | 0.4409 vs 0.2994 — przewaga **1,47×** | `a26` |
| ridge / GBM na POSEIDON (tylko-widmo) | skill **−0,019 / −0,127** — awaria niezależna od klasy modelu | `a26` |
| referencyjny nested sampling, mRMSE / σ | 1.434 / 1.619 dex | `a08` |
| frakcja planet z posteriorem wielomodalnym | 31 – 62 % (per gaz, 200 planet) | `a08` |
| statystyka szumu R: noiseless / 1.0σ / ADC | 0.370 / 1.093 / 0.757 | `a09` |
| MDE (80 %, α=0.05): n=685 / n=64 | **0.0439** / **0.1453** (estymand per-wiersz) | `a12` |
| MDE dla estymandu, który raporty faktycznie cytuja (agregat mRMSE), n=685 | **0.0436** | `a12` |
| liczba seedów w projekcie | 1 (42) | `a12` |
| liczby publikowane bez artefaktu | 4 | `a13` |
| KS vs referencja: ExoBiome / NSF / **prior** / podłoga | 0.7508 / 0.4415 / **0.6101** / 0.0444 | `a24` |
| skill wobec priora na KS: ExoBiome / NSF | **−0.231** / +0.276 | `a24` |
| Wasserstein-1 [dex]: ExoBiome / NSF / podłoga | 1.2293 / 1.0717 / 0.0592 | `a24` |
| martwe importy | 17 w 8 plikach | `a14` |
| sprzężenie T↔gazy / R_p↔gazy / gaz↔gaz (med \|r\|) | 0.221 / 0.107 / 0.053 | `a15` |
| sprzężenie R_p↔T (najsilniejsze w macierzy) | 0.777 | `a15` |
| kurs wymiany T→abundancja (zmierzony, 3315 par) | 0.352 dex/100 K (p90 = 1.000) | `a15.posterior_slope` |
| ten sam kurs z formy zamkniętej, N=7 | 0.333 dex/100 K (iloraz 1.058) | K9(g) |
| wyciek T przez aux na ADC | r = 0.9928, RMSE 48.5 K (sd 401 K) | `a15` |
| przeufność referencyjnego retrievalu na T | σ 38.4 K vs RMSE 201.3 K (5,2×) | `a15` |
| T na crossgen: dostępność / przewidywalność | corr(aux) = +0.0003 / R² = 0.872 | `a15` |

---