# ExoBiome — audyt metodologiczny i inwentaryzacja repozytorium

## 0. Werdykt w skrócie

**Obecny stan wyników jest niepublikowalny, wymaga dużej ilości napraw.** Główne zarzuty:

| # | Fakt | Dowód |
|---|---|---|
| 1 | Wszystkie 685 widm POSEIDON to **linie stałe** (685/685 wierszy `transit_depth_noiseless` ma dokładnie jedną unikalną wartość na 218 binów). Oś główna projektu mierzy reakcję modeli na wejście bez informacji. | `audit/a01` |
| 2 | Porównanie „ExoBiome 0.30 bije SOTA 0.55" wynika z **różnych wejść**. Po zrównaniu konwencji przewaga spada z 1,85× do **1,29×** na ramieniu `median`, które zespół faktycznie opublikował (`holdout_metrics.json: point_estimate = "median"`), a w konwencji zaszumionej **odwraca się** na **1,74×** na korzyść flow. Na ramieniu `mean` odpowiednio 1,35× i 1,76×. | `audit/a03` |
| 3 | Model przewiduje **5 z 7** parametrów swojego benchmarku — pomija **temperaturę i promień**. Pominięta temperatura nie jest słabo sprzężonym dodatkiem, a **dominującą degeneracją** tego, co model raportuje: sprzężenie T–gaz w posteriorze referencyjnym wynosi **0,221** wobec **0,053** gaz–gaz, czyli **4,14× silniej** (663 planety). Na crossgenie temperatura nie jest ani przewidywana, ani odzyskiwalna z tabeli pomocniczej (`corr(T_eq z aux, T prawdziwe) = 0,0003`), choć sama rozciąga skalę wysokości **3,60×** w zakresie 500–1800 K.  | `audit/a15` |


**Część kwantowa jest martwa nie z powodu wyniku, a z powodu konstrukcji:** gałąź „kwantowa" ma 69 434
parametry, z czego **24 są kwantowe** (0,0093 % modelu), jej wyjście to druga głowa klasyczna nad tym samym
kontekstem, do której obwód dokłada 8 z 264 wymiarów wejścia, przemnożoną przez `|tanh(gate)|·scale ∈ [0.0178, 0.0292]`.
Taka architektura nie może wyprodukować **ani** pozytywnej, **ani** rzetelnie negatywnej odpowiedzi na
postawione pytanie badawcze (`audit/a06`).

---

## 0a. Czego dotyczą zarzuty

Większość ustaleń
tego audytu dotyczy danych, pomiaru i sposobu porównania — nie jakości samego modelu. Zarzut wobec aparatury nie jest zarzutem wobec
architektury, a mieszanie ich prowadziłoby do wniosku, że „model jest zły".

### Co model faktycznie robi na osi, która nie jest zepsuta

Zbiór ADC2023 jest **realny i nieuszkodzony**: dane organizatorów challenge'u, nasz split, 4143
wiersze holdoutu, a rekonstrukcja pipeline'u przechodzi z marginesem 11×. Drabina
baseline'ów na tej osi (`a02`, `a26`, przebieg `20260729`):

| klasa | co dokładnie | mRMSE [dex] | skill vs stała treningowa |
|---|---|---:|---:|
| podłoga | stała treningowa (`a02`, cały split treningowy) | 1.4404 | 0 |
| **proste baseline'y** | tylko tabela aux, 8 kolumn | 1.4407 | **−0.0002** |
| **proste baseline'y** | tylko widmo, ridge | 1.0174 | +0.294 |
| **proste baseline'y** | tylko widmo, GBM | 0.4409 | +0.694 |
| baseline organizatorów | CNN z `ADC2023-baseline` | 0.6500 | +0.549 |
| model porównawczy | NSF z rodziny zwycięzcy, `median`, wejście +N(0,σ) — konwencja publikowana | 0.5523 | +0.617 |
| model porównawczy | ten sam NSF, **wejście zrównane** z ExoBiome (widmo jak w pliku) | 0.3846 | **+0.733** |
| **model projektu** | **ExoBiome (skala 1.0)** | **0.2994** | **+0.792** |

**Ridge i GBM w tej drabinie stoją przy 2,8× mniejszym budżecie danych niż ExoBiome, i to zawyża przewagę modelu.** ExoBiome uczył się na całym splicie treningowym ADC (33 138 wierszy). Baseline'y `a26` uczą się na **12 000** — prefiksie listy id (`ids[:limit]`, `a26:315-316`), nie losowej podpróbce, i to jest wartość domyślna `--n-train`, nie dopasowany budżet.

Kierunek błędu jest jednostronny: przy pełnym budżecie ridge i GBM byłyby *nie gorsze*. Przewaga ExoBiome nad nimi jest więc **oszacowaniem górnym**, nie dolnym.

Proste baseline'y to dwa nietuningowane modele ogólnego przeznaczenia, po to, by pokazać, ile z widma
da się wyciągnąć bez sieci: **ridge** to regresja liniowa z karą za duże współczynniki (osiem linii
`numpy`, ta sama receptura co w zespołowym `baseline_smoke.py`), a **GBM** to zespół płytkich drzew
decyzyjnych, który łapie nieliniowości, ale traktuje biny widma jako niezależne kolumny. Dwa różne
learnery zamiast jednego po to, żeby wynik nie zależał od wyboru estymatora.

Wniosek, który wolno postawić: **jako regresor punktowy pięciu abundancji na widmach ADC model działa** —
skill +0,792 wobec stałej treningowej, na 4143 wierszach holdoutu, z rekonstrukcją pipeline'u
zweryfikowaną wiersz po wierszu.

### Dlaczego awaria na crossgenie nie jest zarzutem wobec modelu

Na zbiorze POSEIDON skill wynosi: ExoBiome **−0,111**, noquant **−0,133**, a **ridge −0,019 i GBM
−0,127** dopasowane niezależnie. Wszystkie cztery pomiary są **ujemne**, choć nie równe — rozrzut
między ridge (−0,019) i GBM (−0,127) jest siedmiokrotny. Wspólny jest **znak, nie wielkość**: żadna
z trzech klas uczących się nie bije stałej.

Jednak wszystkie cztery ramiona są **uczone na `tau/train`
i oceniane na POSEIDON**, więc zerowy skill jest zgodny
z **dwiema** hipotezami: dane nie noszą informacji **lub/oraz** żaden model nie przenosi się między
generatorami. Rozdziela to jednak dopiero pomiar
**na samych danych, bez udziału modelu**: 685/685 wierszy
`transit_depth_noiseless` ma jedną wartość na 218 binów (`a01`, kryterium bezprogowe).
Widmo stałe po λ nie może nieść informacji o składzie. Kolejnym punktem byłaby ewaluacja na dobrze wygenerowanych danych,
aby wykluczyć drugą część hipotezy.

Odwrotnie na `tau_val`, gdzie dane mają strukturę: oba warianty ExoBiome biją GBM (1.605, skill +0,444),
choć skromniej niż na ADC — wariant **bez kwantów** 1.423 (+0,507) i wariant kwantowy 1.449 (+0,498),
czyli 1,13× i 1,11×. Lepszy z tej pary jest wariantem klasycznym.

### Podział ustaleń

| dotyczy aparatury pomiarowej | dotyczy samego modelu |
|---|---|
| **K1** widma POSEIDON stałe po λ | **K5** faza wspólna 2 z ~24 epok; wysłany checkpoint sprzed odmrożenia backbone'u |
| **K11** Nasza implementacja generatora TauREx bez CIA | **K6** 24 z 69 434 parametrów gałęzi są kwantowe (0,0093 %) |
| **K3** benchmark porównuje dwie konwencje wejścia | **K7** bramka zero-init, niezbieżna, CO₂ o przeciwnym znaku |
| **K4** metryki raportowane w punkcie pracy, w którym model nie był walidowany | **K9** 5 z 7 parametrów benchmarku |
| **K10** metryka projektu ≠ metryka benchmarku| |


### Co zostało zrobione dobrze

1. **Dwuetapowy trening z klasycznym punktem startowym i faza kwantowa.**
   `config.json` flagowego runu wskazuje `init_checkpoint_path → stage1_classical/best_model.pt`, a gałąź
   kwantowa dochodzi w etapie 2 z zamrożonym backbone'em (`quantum_backbone_freeze_epochs = 6`). To jest
   właściwy szkielet ablacji. Co więcej, ta faza
   **wykonała się w całości i działała**: przez wszystkie sześć zamrożonych epok walidacja poprawiała się
   monotonicznie (0,29333 → 0,29081, `history.csv`). Zarzut „gałąź kwantowa nie zdążyła się wytrenować"
   jest **wykluczony**. Wada K6/K7 nie leży ani w projekcie eksperymentu, ani w przebiegu tej fazy, tylko
   w tym, że wkład jest o dwa rzędy wielkości za mały, by cokolwiek zmierzyć. Osobno (problem z K5) faza **wspólna** (backbone odmrożony) dostała 2 epoki z ~24 zaplanowanych i destabilizowała się
   natychmiast (+13,0 % na epoce 7, zbieżnie z `backbone_frozen 1 → 0`; w tej samej epoce rośnie też
   `quantum_scale`, więc przyczyny są nierozdzielne — patrz K5), a wysłany checkpoint
   jest **sprzed niej**.
2. **Osobny wariant bez kwantów** (`models/taurex_exobiome_without_quant`)
3. **Reimplementacja modelu odniesienia z rodziny zwycięzcy ADC2023**, wytrenowana lokalnie na naszym
   splicie, z zapisanym checkpointem i metrykami, nie zacytowana z pracy.
4. **Determinizm modelu.** Dwa wywołania na tym samym checkpoincie dają `max |p₁ − p₂| = 0.000e+00`.
   Jedyne wcześniej niepowtarzalne wielkości, czyli estymator punktowy NSF, losowany z posterioru flow
   (`a24:189` — seed, `a24:195` — losowanie), są już zaseedowane po stronie audytu i potwierdzone bit-identyczne między
   przebiegami.

### Na czym model był trenowany

- **Pewne, z kodu:** trening ExoBiome **nie zawiera żadnej augmentacji szumem**. Skan całej ścieżki
  `models/ariel_exobiome/*.py` po `randn`, `np.random`, `torch.normal`, `augment`, `add_noise` daje
  **dwa** trafienia i **żadne** nie jest perturbacją widma: `np.random.seed(seed)` w `training.py:127`
  (seed) oraz `torch.randn(3 * self.n_qubits * self.num_blocks, ...)` w `model.py:185` (inicjalizacja
  wag obwodu — to z niej pochodzi `circuit_init_scale = 0.1` cytowane w K7). Ścieżek próbkujących szum
  **wejścia** nie ma ani jednej. `config.json` flagowego runu nie ma żadnego parametru szumu. Model konsumuje widmo tak, jak leży w pliku, a wektor σ dostaje jako
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

Kod audytowy leży w `audit/`. Każdy check ładuje wyłącznie zamrożone artefakty przebiegów, nie modyfikuje kodu
modeli ani treningu, i zapisuje samoopisujący się rekord JSON (commit gita, wersje środowiska, sha256 wejść,
werdykt) do `reports/audit/<data>/`.

```bash
cd ../hack4sages
./.venv-qml/bin/python audit/run_all.py
```

```bash
./.venv-qml/bin/python audit/run_all.py --fast
```

Wymagane wejścia: same zbiory nie są śledzone na `main` (`.gitignore:23,25,40` — `data/ariel-ml-dataset/`,
`data/val_dataset/`, `data/generated-data/`), więc oś ADC audytu wymaga podłączenia worktree
`origin/iwosmu/data-artifacts` (`docs/INWENTARYZACJA_DANYCH.md` §2) i zbudowania cache'u
`data/generated-data/ariel_winner_nf_prepared`. Bez niego `a03` emituje `INFO` z polem
`baseline_nsf.error`, nie `FAIL` (`a03:110-115,179`).

`run_all.py` kończy się kodem ≠ 0, jeśli którykolwiek check ma status FAIL. Pełna specyfikacja checków,
jest w `audit/README.md`.

**Walidacja kodu audytu.** Liczby w K3, K4 i K9 nie są odczytane z artefaktów z repo, musiały być przeliczone
rekonstrukcją pipeline'u ExoBiome w `audit_lib.py`, która od nowa czyta surowy HDF5, stosuje zapisane
skalery i przepuszcza dane przez model. Gdyby ta rekonstrukcja różniła się od oryginału, wszystkie trzy
ustalenia byłyby bezwartościowe, dlatego jest sprawdzana osobnym checkiem. `a27_pipeline_fidelity`
porównuje ją z zacommitowanymi predykcjami **wiersz po wierszu**, na wszystkich 4143 wierszach × 5 gazach.

**Tolerancja `1e-4` (`a27:46`) jest przyjęta, nie wyprowadzona, i ma sens tylko wobec jednej z dwóch referencji, które `a27` sprawdza.**

| | `mac_cpu` (ta sama ścieżka kodu) | `gpu_amp` (CUDA + bf16, artefakt opublikowany) |
|---|---:|---:|
| tolerancja PASS | 1,0e-4 | 1,0e-4 |
| max różnicy (20 715 komórek) | 8,9e-6 | 9,6e-1 |
 wierszy nad tolerancją | 0 z 4143 | 4143 z 4143 |

`mac_cpu` to rekonstrukcja na tej samej ścieżce co audyt (CPU, fp32) — tam różnica to szum zaokrągleń, więc 1e-4 ma **11× zapasu**. `gpu_amp` to zacommitowany plik, który projekt faktycznie opublikował (CUDA + bfloat16) — na nim żadna komórka nie mieści się w tolerancji.

To nie jest błąd rekonstrukcji, tylko właściwość ścieżki numerycznej AMP. Ale ma konsekwencję: **status `PASS` liczy się wyłącznie z kolumny `mac_cpu`** (`a27:132-135`); `gpu_amp` jest tylko raportowane. `a27` waliduje re-ewaluację na CPU, nie artefakt, który trafił do publikacji. „Margines 11×" dotyczy pierwszego, nie drugiego.


### 1.1 Reprodukowalność harnessu

Pełny przebieg `run_all.py` (`20260729`): **21 checków**, jeden katalog
(`reports/audit/20260729/`), `is_partial_run: False`, **0 ERROR**, wynik
`FAIL 17 / WARN 1 / INFO 1 / PASS 2`.

**Harness w pełni odtwarzalny.** Jedyna wcześniej niepowtarzalna wielkość (estymator punktowy NSF)
brała się z niezaseedowanego losowania posterioru (`IndependentNSF.sample`, `model.py:54`, bez
generatora). Naprawione w audycie: `a03`/`a24` seedują globalny RNG (`POSTERIOR_SAMPLE_SEED = 42`,
zapisane w payloadzie) przed każdym losowaniem, bez ingerencji w kod zespołu.

---

## 2. Tabela inwentaryzacyjna

Status na przebiegu `reports/audit/20260729/` (21 checkow, `is_partial_run: False`, **0 ERROR**).

| id | problem | dowód | wynik checku |
|---|---|---|---|
| K1 | Widma POSEIDON są stałe po długości fali | `a01_spectral_variation` | FAIL |
| K1(b) | Smoke baseline zespołu **był uruchomiony**, ustalenie postawił zarchiwizowany audyt | `a29_smoke_baseline_recovery` | PASS (ustalenie zawężone — patrz K1(b)) |
| K1(c) | Diagnoza: skrypt do generacji atmosfery poprawny. |  `d01` stage 1 | zawężone, decyzja 2026-07-27 |
| K3 | Niezgodne konwencje wejścia w benchmarku vs SOTA + niedotrenowany model zwycięzców |  `a03_input_convention`, `a05`, `a09` | FAIL |
| K4 | Raportowane metryki przy `quantum_scale=1.0`, selekcja przy 0.5 | `a04_quantum_scale_provenance` | FAIL |
| K5 | Flagowy checkpoint z epoki 6; faza wspólna przerwana ręcznie w 8/30 epok | `a05_training_completeness` | FAIL |
| K6 | Wkład kwantowy nieodseparowalny od 69 410 parametrów klasycznych | `a06_param_accounting` | FAIL |
| K7 | Model sam wyciszył kwant: bramka jest zero-init, a jej wartość końcowa niezerowa |  `a07_gate_dynamics` | FAIL |
| K9 | Model przewiduje 5 z 7 parametrów benchmarku; T sprzężona z abundancjami 4,14× silniej niż gazy między sobą | `a15_target_completeness` | FAIL |
| K11 | Nasza konfiguracja TauREx generuje bez CIA: brak głównego kontinuum w H₂/He | analiza kodu, m. in. `data/crossgen_biosignatures/taurex_backend.py:154-157` | — |
| K10 | Raportowana metryka (mRMSE) **nie jest** metryką ADC2023 (0,8·KS na 7 marginalnych + 0,2·widmowa); model punktowy jest na niej niepunktowalny | literatura + `a24_official_metrics` | FAIL |

Dodatkowe problemy, nierozwinięte w osobnych sekcjach:

| problem | dowód | wynik checku |
|---|---|---|
| `taurex_fmpe`: holdout ≡ validation | `a10_split_integrity` | FAIL |
| Brak zbioru testowego in-domain na TauREx | `a10_split_integrity` | FAIL |
| Jeden seed w całym projekcie (42); brak korekty na wielokrotność | `a12_significance_power` | FAIL |
| Generatory różnią się opacity/rozdzielczością/CIA/He | publikacje | — |

---

## 3. Problemy krytyczne — dowody

### K1. Widma POSEIDON są stałe po długości fali

**Gdzie znaleziony.** `data/TauREx set/spectra.h5`, pole `transit_depth_noiseless`, wszystkie 685
wierszy POSEIDON. Modele czytają `transit_depth_noisy` (`models/taurex_exobiome/dataset.py:287`).
Zdania, które to ustalenie podważa, stoją w `reports/taurex_model_comparison.md` (cały ranking
transferu). Kod generatora:
`poseidon_backend.py:78-88` — `from POSEIDON.core import compute_spectrum, make_atmosphere, ...`, czyli
import prawdziwej biblioteki, nie reimplementacja fizyki — oraz `poseidon_backend.py:50`, gdzie
powstaje siatka ciśnień `np.geomspace(PRESSURE_MIN_BAR, PRESSURE_MAX_BAR, PRESSURE_LEVELS)`.
Hipoteza wtórna dotyczy `data/scripts/bootstrap_poseidon_input_data.sh:10`.

**Wyjaśnienie.**

```
poseidon  transit_depth_noiseless  relvar_med=0.000e+00  const rows=685/685  feat/sigma=0.000  frac SNR>1=0.000
poseidon  transit_depth_noisy      relvar_med=2.144e-03  const rows=  0/685  feat/sigma=0.998  frac SNR>1=0.488
tau       transit_depth_noiseless  relvar_med=1.289e-02  const rows=0/41423 feat/sigma=6.417  frac SNR>1=0.922
tau       transit_depth_noisy      relvar_med=1.394e-02  const rows=0/41423 feat/sigma=6.488  frac SNR>1=0.997
```

`transit_depth_noiseless` dla POSEIDON: **jedna skalarna głębokość tranzytu powtórzona 218 razy** (wiersz 0:
`0.01698546` × 218; `n_unique_values_median = 1`, `bit_constant_rows = 685/685`). Widmo, które nie zależy
od długości fali, nie może nieść informacji o składzie atmosfery, niezależnie od przyczyny.


A dlaczego tak? **Mechanizm maskujący w preprocessingu.** Zabezpieczenie, które tego nie łapie: `data/crossgen_biosignatures/validate_dataset.py:78-90`. Po prostu **pipeline modelu nie mógł tego zgłosić, bo usuwa dokładnie tę
informację w pierwszym kroku**. ExoBiome dzieli każde widmo przez jego własną średnią
(`models/ariel_exobiome/dataset.py:414-416`, ten sam kod w `models/taurex_exobiome/dataset.py`);
reimplementacja zwycięzcy odejmuje średnią, dzieli przez odchylenie z `clamp(min=1e-6)` i podaje średnią
oraz odchylenie jako dwie osobne cechy (`models/adc_winner_on_ariel/preprocessing.py:151-155`). W obu
wariantach człon stały wypada z wektora widmowego, więc widmo płaskie **po normalizacji jest
nierozróżnialne skalą od zdrowego**: zostaje 218 kanałów (dla ADC 52) szumu o poprawnej amplitudzie.

**Zakres tego ustalenia.** Generator jest nakładką na prawdziwą bibliotekę (`from POSEIDON.core
import …`), więc defekt musi leżeć w sposobie wywołania API albo w torze danych opacity.

Ważne: **Nie jest to błąd kształtu
zwracanej tablicy.** Obie ścieżki rebinningu by go złapały: `rebin_spectrum` waliduje `values.shape`
jawnie, a `fixed_native_matrix.dot(...)` rzuciłby wyjątek na niezgodności wymiarów.

**Skąd próg `1e-4` i czy jest wystarczający.** Dla tego werdyktu **nie ma znaczenia**: decyduje kryterium
bezprogowe („zero wierszy bit-stałych"), a POSEIDON ma 685/685 wierszy bit-stałych przy
`rel_variation = 0.000e+00` **dokładnie**.

Jednak od tego próg da się wyprowadzić fizycznie, a nie tylko przyjąć: relatywna amplituda cechy transmisyjnej na jedną skalę wysokości wynosi `2H/R_p`,
gdzie `H = kT/(μg)` (Line et. al 2016, arXiv:1511.09443). Policzone z etykiet tych 685 planet (`temperature_k`, `log_g_cgs`,
`planet_radius_rjup`, μ z `vmr_h2`/`vmr_he`): μ mediana **2,314**, `H` mediana **208 km** (zakres 38–974),
a `2H/R_p` mediana **5,42e-3** przy **minimum 9,48e-4** na całym zbiorze. Próg leży więc **9,5× poniżej
najsłabszego fizycznie możliwego sygnału** w tym zbiorze, cokolwiek prawdziwego musi go przekroczyć.
Zmierzone tau ma `p01 = 1,99e-3`, czyli 20× nad progiem; POSEIDON **0,0**.

Ograniczenie samego checku: `a01` nie wykrywa *struktury o złej
amplitudzie*. Generator z brakującym członem opacity (ustalenie K11) przechodzi ten check z definicji.

Literatura. TauREx I i III (arXiv:1409.2312, arXiv:1912.07759v2) są referencją dla tego, jak powinna
wyglądać zgodność dwóch generatorów (CIA, ciśnienie referencyjne, listy linii) — bez niej „gap" mierzy
różnicę konfiguracji, nie fizyki. Ramy teoretycznej dla gapu jako misspecyfikacji (arXiv:2210.06564,
RNPE) wolno użyć **dopiero po** naprawie K1: misspecyfikacja wymaga dwóch poprawnych modeli, nie
jednego poprawnego i jednego zwracającego stałą.

Dlaczego dyskwalifikuje: unieważnia każdą liczbę cross-generator w repo (3.2156, 3.2796, 3.4531, 2.8946,
wszystkie „gapy", cały `reports/taurex_model_comparison.md`). Awaria procesowa: `data/crossgen_biosignatures/validate_dataset.py:78-90` sprawdza kształt, skończoność,
dodatniość, liczności per generator (`:87-90`), zakresy priorów (`:92-101`) i prewalencję (`:122-127`) — nie sprawdza, czy widmo zależy od długości fali.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a01_spectral_variation.py` |
| komenda | `./.venv-qml/bin/python audit/a01_spectral_variation.py` |
| kryterium PASS | zero wierszy bit-stałych ∧ mediana feature/σ > 1 ∧ mediana `std_bins/\|mean_bins\|` > 1e-4; sprawdzany jest **każdy** wiersz |
| wartość oczekiwana | Status `FAIL`, `failing_generators = [poseidon]`. POSEIDON `transit_depth_noiseless`: `bit_constant_rows = 685/685`, `rel_variation_median = 0.0`, `feature_amplitude_over_sigma_median = 0.0`, `first_row_head = 0.016985462978482246`; `transit_depth_noisy`: `apparent_amplitude_over_sigma_median = 0.998`, `frac_rows_snr_gt_1 = 0.488`, `frac_rows_snr_gt_3 = 0.0`. tau, wszystkie 41 423 wiersze, to samo pole: `6.488` / `0.997` / `0.721` |
| co by to obaliło | choćby dwie różne wartości w wierszu `transit_depth_noiseless` przy medianie feature/σ > 1 — czyli istotna zmienność po λ przekraczająca szum. Wtedy oś cross-generator byłaby ważna i wszystkie liczby POSEIDON wracają. Ustalenia **nie** obala 218 unikalnych wartości w `transit_depth_noisy` (to szum, `feat/sigma = 0.998`) ani ustalenie przyczyny (K1(c)) |
| rekord | `reports/audit/20260729/a01_spectral_variation.json` |

### K1(b). Smoke baseline zespołu zostawił ślad, który wskazywał na POSEIDON-a

**Gdzie znaleziony.** `data/TauREx set/baseline_smoke.json` wraz z `data/TauREx set/baseline_poseidon_predictions.csv`. **Nieśledzone w gicie w chwili przekazania; do indeksu wprowadził je ten audyt** (`4c431db`, 2026-07-27 22:09, po dodaniu negacji w `.gitignore:46-50` w `61adef1`).

**Wyjaśnienie.**

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
`U(−12, −2)`. Nie są literalnie stałe — każda z pięciu kolumn ma 685 unikalnych wartości, wszystkie
różne — ale wszystkie mieszczą się w pasmie **[−7,527; −6,739]** o szerokości 0,788 dex, czyli
**0,27 σ** etykiet (σ = 2,894). Predyktor zapadł się więc do wąskiego pasma wokół średniej prioru,
ignorując wejście.

Sygnał 3 — dokładność binarna równa prewalencji klas, co do czwartego miejsca po przecinku.

| gaz | `test_binary_accuracy` | prewalencja z `manifest.json` | różnica | `val_binary_accuracy` | prewalencja tau | różnica |
|---|---:|---:|---:|---:|---:|---:|
| H₂O | 0,5766 | 0,5766 | **+0,0000** | 0,6357 | 0,6034 | +0,032 |
| CO₂ | 0,5956 | 0,5956 | **+0,0000** | 0,6463 | 0,6037 | +0,043 |
| CO | 0,6234 | 0,6234 | **+0,0000** | 0,5987 | 0,5988 | −0,000 |
| CH₄ | 0,6088 | 0,6088 | **+0,0000** | 0,6564 | 0,6006 | +0,056 |
| NH₃ | 0,6058 | 0,6058 | **+0,0000** | 0,6393 | 0,5962 | +0,043 |

Mechanizm wynika z sygnału 2: całe pasmo predykcji **[−7,527; −6,739]** leży **powyżej** progu obecności
`−8,0`, więc **każdy** z 685 wierszy dostaje dla każdego gazu etykietę „obecny". Dokładność klasyfikatora,
który zawsze orzeka ze jest, jest równa udziałowi wierszy, w których gaz faktycznie jest czyli
prewalencji. Sygnał 3 jest bardziej sprawdzeniem sygnału 2, a nie osobnym dowodem.

Sygnał 4 — metadane potwierdzają, że to ten bundel: `feature_dim = 219` (218 binów + `sigma_ppm`),
`train_rows = 37281`, `val_rows = 4142`, `test_rows = 685`.

**Wniosek** Oba artefakty są
zadeklarowane w **kodzie produkcyjnym** jako pola `DatasetPaths`
(`data/crossgen_biosignatures/constants.py:88-89`, śledzone od 2026-03-11), a `a29` znajduje **osiem**
konsumentów odczytujących je po ścieżce, wszystkie śledzone w gicie i starsze od tego audytu. Do tego
zarchiwizowana próba audytu zespołu (`archive/…/11_baseline_comparison.md`) opisuje ten artefakt jako
mean-predictor i oznacza go jako CRITICAL. Mimo tego ani `reports/taurex_model_comparison.md`, ani plan
prac nie zawierają śladu tej diagnozy.


**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a29_smoke_baseline_recovery.py` weryfikacja przez lekturę `data/TauREx set/baseline_smoke.json`, `data/TauREx set/baseline_poseidon_predictions.csv` i `data/TauREx set/manifest.json` |
| komenda | `./.venv-qml/bin/python audit/a29_smoke_baseline_recovery.py` |
| kryterium PASS | Kryterium jest odwrócone, bo ten check ma **szansę obalić** ustalenie, a nie je potwierdzić. **PASS = ustalenie obalone**, jeśli zachodzi **którykolwiek** z czterech warunków; **FAIL = ustalenie stoi**, gdy nie zachodzi żaden. Warunki poniżej |
| ↳ (a) | brakuje któregoś artefaktu w `data/TauREx set/`, czyli nie ma czego czytać |
| ↳ (b) | CSV nie odtwarza `test_rmse` z JSON-a, per gaz, w granicy `1e-9` → artefakty nie pochodzą z jednego przebiegu na tym bundlu |
| ↳ (c) | brak kontrastu znaków: `sign(skill[tau_val]) == sign(skill[poseidon_test])` → dostarczone liczby same z siebie nie wskazywały POSEIDON-a |
| ↳ (d) | dowolny plik poza `data/TauREx set/`, `audit/`, `docs/` i `reports/audit/` odczytuje artefakty po nazwie - teza nikt na to nie spojrzał nie może przejść |
| wartość oczekiwana (**z payloadu `a29`**) | `t1_artefact_consistency`: `max_abs_diff = 1.8e-15` przy `tolerance = 1e-9`, `n_rows_csv = 685`, wszystkie `sample_id` dopasowane; `t2_mrmse.poseidon_test.mrmse = 2.8954` (per gaz `2.8738 / 2.9661 / 2.8482 / 2.9477 / 2.8410`, `mrmse_recomputed_from_csv` zgodne) i `t2_mrmse.taurex_val.mrmse = 2.6147` (`independently_recomputable = false` — bundel nie ma predykcji dla splitu tau); `t3_skill`: wobec baseline'u stałego 2,8940 skill **−0,0005** na POSEIDON i wobec 2,8852 skill **+0,0938** na tau/val; `t4_prediction_spread.mean_ratio = 0.0221`, `mean_prediction_all_gases = −7.0141` przy `prior_mean = −7.0`; `t5_collapse_to_prior_mean.abs_diff = 0.0016`; `t6_consumption.n_consumers = 8`, wszystkie `git_tracked = true`, wszystkie starsze od audytu |
| wartość oczekiwana  | Bezpośredni odczyt `data/TauREx set/baseline_smoke.json` i `manifest.json`: `test_binary_accuracy` = prewalencja z `manifest.json: generator_summary.poseidon.prevalence` co do czwartego miejsca dla wszystkich pięciu gazów (0,5766 / 0,5956 / 0,6234 / 0,6088 / 0,6058); `feature_dim = 219`, `train_rows = 37281`, `val_rows = 4142`, `test_rows = 685`; pasmo predykcji `[−7,527; −6,739]` policzone z `baseline_poseidon_predictions.csv` |
| co by to obaliło | każdy z (a)–(d) z osobna. W tym przebiegu zaszło **wyłącznie (d)**, więc `a29` = `PASS`. (a)–(c) nie zaszły: artefakty są kompletne, spójne między sobą i wykazują kontrast znaku skill między `taurex_val` a `poseidon_test`. Obalony jest tylko fragment o nieodczytaniu wyniku, dlatego ustalenie „nie znalazło odbicia w wynikach". |
| rekord | `reports/audit/20260729/a29_smoke_baseline_recovery.json` |

---

### K1(c). Diagnoza Etapu 1: atmosfera jest zdrowa, opacity jest **nasycone**

**Gdzie znaleziony.** `poseidon_backend.py:50` (siatka ciśnień) i `poseidon_backend.py:80-88`
(wywołania `make_atmosphere` / `compute_spectrum`), zdiagnozowane przez
`audit/d01_poseidon_diagnosis.py --stage 1`, próbka `poseidon_000001`
(T = 998,7 K, R_p = 0,979 R_jup, log g = 3,27), na POSEIDON **1.3.2** — wersji, która wygenerowała
zbiór. Etap 1 nie wymaga bazy opacity. Wersję generatora podaje `data/TauREx set/manifest.json`.

**Wyjaśnienie.** Wszystko, co Etap 1 potrafi sprawdzić, jest poprawne:

| sprawdzenie | wynik |
|---|---|
| mieszaniny w atmosferze vs żądane | `X` (7, 100, 1, 1); ślady −10,172 / −8,054 / −5,576 / −5,568 / −10,973 — **identyczne z etykietami** |
| tło H2/He | 0,850 / 0,150 — czyli `He_fraction = 0,17647` jako He/H2 daje zamierzone proporcje |
| profil temperatury | jedna unikalna wartość (izoterma), zgodnie z `PT_profile="isotherm"` |
| promień referencyjny | warstwa najbliższa 10 bar ma `r = 69 982,4 km` = **dokładnie** przekazane `R_p_ref` |
| monotoniczność | `P[0]=1e-6 bar` → `r=73 249,6 km`; `P[-1]=100 bar` → `r=69 554,5 km` — poprawnie |
| skala wysokości | mediana `H` = 200,46 km (zakres 190,48–211,25; 192,83 przy `P_ref` — `H` **nie** jest stałe pod izotermą, bo `g` maleje jak 1/r² na 18 skalach); rozciągłość 3695 km, czyli 18,43 H |
| **kierunek siatki ciśnień** | **bez znaczenia**, POSEIDON sortuje wewnętrznie |


Płaskie widmo ma dwie możliwe przyczyny, dające ten sam objaw z przeciwnych powodów: **opacity zerowe**
(atmosfera nie pochłania nic, więc światło blokuje wyłącznie ciało stałe planety) albo **opacity
nasycone** (atmosfera pochłania tak silnie, że τ=1 jest osiągane już w jej najwyższej, najcieńszej
warstwie). Test rozstrzyga między nimi,
sprawdzając, **któremu promieniowi fizycznemu odpowiada zapisana głębokość tranzytu** — bo obie
hipotezy przewidują inny promień efektywny. Promień wyliczony z zapisanej głębokości
pasuje do promienia **szczytu atmosfery** i wyraźnie odbiega od dna
atmosfery oraz od `R_p_ref`. Zgodność ze szczytem wskazywałoby na **nasycenie**.

Test rozróżniający, który wskazał przyczynę. Zapisana płaska głębokość dla tej próbki to
0,01698546. Promień, który ją implikuje, to **73 268,3 km**. Porównanie z trzema kandydatami:

| poziom | r [km] | (r/R_s)^2 | iloraz do zapisanej |
|---|---:|---:|---:|
| dno atmosfery (100 bar) | 69 554,5 | 0,01530721 | 0,901 |
| R_p_ref (10 bar) | 69 982,4 | 0,01549613 | 0,912 |
| **szczyt atmosfery (1e-6 bar)** | **73 249,6** | **0,01697680** | **0,9995** |

Zgodność do **0,05 %**; różnica 19 km to 0,1 skali wysokości.

Teraz "**dlaczego ekstynkcja nasyca**". Kandydaci w kolejności prawdopodobieństwa: niezgodność bazy
opacity (`opacity_database="High-T"`, `database_version="1.3"`, przy czym bootstrap sprawdza obecność
bazy *standardowej*, więc mogła zostać wczytana niewłaściwa tabela), błąd jednostek w tablicach
przekrojów czynnych, albo zachowanie `opacity_sampling` poza siatką `T_fine`/`log_P_fine`.

DECYZJA PROJEKTOWA (2026-07-27): przyczyna zostaje zamknięta jako ZAWĘŻONA. Etap 2 wymaga 72,1 GB
danych wejściowych. Przechodzi on do
backlogu jako warunek wstępny (regeneracja danych).

**Wersja przypięta i zweryfikowana.** Zbiór wygenerowano POSEIDON-em **1.3.2** (`manifest.json`).
Diagnozę Etapu 1 wykonano na wersji `v1.3.2`.

Literatura. Lista elementów konfiguracji decydujących o amplitudzie cech (CIA, ciśnienie referencyjne,
listy linii; arXiv:1409.2312) to lista, którą Etap 2 musi przejść pozycja po pozycji. Welbanks
& Madhusudhan 2019 (arXiv:1904.05356) pokazują, że wolno zamrozić `R_ref` **albo** `P_ref`, nie oba,
i że CIA jest niezbędne — a skoro Etap 1 potwierdza, że `R_p_ref` i `P_ref` są przekazane poprawnie
(warstwa 10 bar odtwarza `R_p_ref` dokładnie), kandydat „zła konwencja promienia referencyjnego"
wypada i zostaje sam tor przekrojów czynnych.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/d01_poseidon_diagnosis.py --stage 1`|
| komenda | POSEIDON-a nie ma w `.venv-qml` ani `.venv-cnn`, więc Etapy 1 i 2 idą z osobnego interpretera; do `run_all.SUITE` wchodzi wyłącznie `d01 --stage 0` (`run_all.py:71`, stąd rekord `INFO` w przebiegu `20260729`), a rekord poniżej pochodzi z ręcznego Etapu 1. Pełna instalacja na przypiętej wersji generującej 1.3.2 oraz wywołanie z `-u` — w `audit/README.md`. Skrót: `MPLCONFIGDIR=/tmp/mpl <venv>/bin/python -u audit/d01_poseidon_diagnosis.py --stage 1` |
| kryterium PASS | dokładne wywołanie z repo odtwarza widmo o tej samej zmienności relatywnej co tau (~1e-2). Etap 1 tego kryterium **nie może** spełnić, bo do widma potrzebna jest baza opacity. Status rekordu to jednak `FAIL`, nie `INFO`: `saturation_test` rozstrzyga znak defektu bez żadnych danych opacity, a nasycenie jest defektem |
| wartość oczekiwana | Wszystkie liczby geometrii i testu wyżej pochodzą z wersji **1.3.2** generującej zbiór, czyli właściwej. Dodatkowo: `input_data_root = null`, `opacity_files = []` (blocker Etapu 0 — Etap 2 potrzebuje zenodo 16107813, 72,1 GB); `grids_bit_identical = true`; `verdict_H1` niepotwierdzona. Rdzeń dowodu, `saturation_test`: `recorded_depth = 0,016985462978482246` (`n_unique_bins = 1`), `R_s_km = 562 182,28`, `implied_radius_km = 73 268,254`; ilorazy `atmosphere_bottom = 0,9012`, `R_p_ref = 0,9123`, `atmosphere_top = 0,9995` przy `tolerance = 5e-3` → `matching_candidates = [atmosphere_top]`, `verdict_opacity = SATURATED` |
| co by to obaliło | **przyczyna jest zawężona, nie ustalona**. Zawężenie upada, gdyby: odwrócenie siatki ciśnień zmieniało wynik (nie zmienia — bit-identyczny na obu wersjach), albo gdyby `(r_szczyt/R_s)²` nie zgadzało się z zapisaną głębokością do 0,05 %, albo gdyby Etap 2 pokazał poprawnie wczytane przekroje czynne i powierzchnię tau = 1 **nie** przy 1e-6 bar. W takim wypadku płaskość ma źródło poza torem opacity. |
| rekord | `reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json` (**1.3.2, wersja generująca**), `timestamp_utc` 2026-07-29T17:39:54Z. Werdykt `SATURATED` stoi na tym jednym rekordzie |
---

### K3. Benchmark vs SOTA porównuje różne wejścia

**Gdzie znaleziony.** Sporne twierdzenie — publikowana przewaga 1,85× — stoi w raportach porównawczych
ADC; split potwierdza `saved_split_manifest.json`. Rozjazd konwencji jest wypalony w kodzie treningowym
na main, w pięciu miejscach opisanych niżej. Wniosek: **po stronie NSF przełącznik szumu
istnieje i jest zaszyty na `True` w każdym wywołaniu, a po stronie ExoBiome takiego przełącznika nie ma
w ogóle**.

**(a) NSF: wtrysk szumu i jego pięć wywołań.** Sam wtrysk,
`models/adc_winner_on_ariel/preprocessing.py:148-149`:

```python
    if sample_noise:
        sampled_spectra = torch.normal(mean=spectra, std=noise, generator=noise_generator)
```

`train.py` woła to z `sample_noise=True` **bez żadnej flagi konfiguracyjnej** w pięciu punktach:
`:161` (pętla treningowa), `:181` (walidacyjny NLL), `:219` (walidacyjny mRMSE per epoka), `:253`
(finalna walidacja), `:267` (finalny holdout) — pełny `grep` po `sample_noise` w tym pliku daje poza
tym tylko sygnaturę `:71` i jej przekazanie `:85`. Rozstrzygające jest `:267`, bo tam konwencja
zaszumiona trafia **do artefaktu**, z którego `a02` czyta 0.5523: wywołanie `evaluate_point_metric(…,
sample_noise=True, …)` jest bezpośrednio nad `save_metrics(run_dir / "holdout_metrics.json", …)`
(`:271-272`). Czyli szum nie jest dodawany dopiero przez `scripts/reeval_sota.py` — reeval tylko
powtarza konwencję, w której powstał `holdout_metrics.json`.

**(b) ExoBiome nie ma czego wyłączyć.** `instrument_noise` wchodzi jako **statyczny kanał
wejściowy**, nie jako σ do losowania: `SAMPLE_SPECTRAL_CHANNELS`
(`models/ariel_exobiome/constants.py:44-47`) wymienia `instrument_spectrum` i `instrument_noise` jako
dwa kanały danych. Model więc *widzi* wektor szumu (`MODEL_SPECTRAL_CHANNELS`, `:54-59`), ale samo
widmo nigdy nie jest perturbowane. Sygnatura `evaluate_labeled_split`
(`models/ariel_exobiome/training.py:238-246`) nie ma żadnego parametru szumu, a jej pełna lista to
`(model, split, target_scaler, batch_size, loss_fn, enable_quantum=True, quantum_scale=1.0)` —
a `gather_labeled_batch` (`:255`) podaje zapisane spektra prosto do `model(...)`.

Potwierdzenie negatywne: w `models/ariel_exobiome/dataset.py` szukanie `noise|rng|random|jitter|perturb`
trafia wyłącznie w `random_state=seed` (`:467`) i `random_state=seed + 1` (`:478`), czyli w podział
zbioru. Ani jednej ścieżki próbkującej szum.

**(c) Kryterium stopu ≠ metryka raportowana** (kodowe potwierdzenie „niedotrenowanego baseline'u").
W pętli `models/adc_winner_on_ariel/train.py` trzy rzeczy dzieją się w trzech różnych miejscach:

| co | gdzie | na czym |
|---|---|---|
| licznik cierpliwości rośnie | `:197-202` | **wyłącznie** w gałęzi `if val_nll < state.best_val_nll: … else: epochs_since_improvement += 1` |
| `best_model_by_mrmse.pt` jest zapisywany | `:228-230` | tylko wewnątrz `if metric_every > 0 and epoch % metric_every == 0`, czyli co kilka epok |
| trening jest przerywany | `:236-238` | `if state.epochs_since_improvement >= patience: break` — czyli na cierpliwości **NLL** |

**Model porównawczy (NSF) zatrzymano metryką inną niż ta, na której go porównujemy.**

Trening zatrzymał się po 79 z 300 epok, sterowany cierpliwością na `val NLL`. Metryka porównania to mRMSE, mierzona co 10 epok: `0,6299 / 0,6122 / 0,5736 / 0,5799 / 0,5629 / 0,5450 / 0,5657`. Minimum wypada na pomiarze 6 z 7 (epoka 60); jedyny późniejszy pomiar jest o 3,8% gorszy (epoka 70).

Pole `comparison_metric_still_improving = true` w `a05` mogłoby sugerować, że mRMSE dalej by spadało. Pole sprawdza tylko, czy minimum wypadło w dwóch ostatnich pomiarach (`a05:126`, stała `IMPROVING_TAIL_MEASUREMENTS = 2`, `a05:48`), nie czy trend jest malejący. To nie jest wniosek metryka jeszcze by spadała, tylko bardziej model zatrzymano na kryterium innym niż porównawcze, a jedyny pomiar po minimum jest gorszy, więc zbieżności mRMSE nie da się orzec w żadną stronę.


**(d) Ta sama asymetria obowiązuje w ramieniu cross-generator**. Identyczny kod i numery linii w przeniesionych
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

**Ile szumu jest w widmach ADC?** Dokumentacja zbioru odpowiada na to wprost: widma są idealne, szum nie jest do nich wliczony, a tablica σ jest dostarczana osobno (arXiv:2309.09337, §1.2 Noise; potwierdzone w publicznym repozytorium AstroAI-CfA).

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
| ↳ zastrzeżenie do `a09` | jego `FAIL` nie jest rozstrzygający i nie wchodzi do żadnego wniosku K3: próg to `0,80 × kalibracja tau_noisy` (`a09:34`) — ułamek przyjęty, przy `0,70` ADC przechodzi — a kalibracja idzie z 218 binów ze skalarną σ na 52 biny z σ per-bin. Do tego `pass_criterion` rekordu mówi „R ≥ 1", a kod porównuje z `0,8744` (`a09:100,109`) |
| komenda | `./.venv-qml/bin/python audit/a03_input_convention.py --split holdout --seeds 42,1,2` |
| kryterium PASS | `sign(ExoBiome − baseline)` niezmienny wobec konwencji wejścia |
| wartość oczekiwana | Status `FAIL`, `sign_flips = true`. ExoBiome: czysty `0.298693` (skala 1.0, raportowana — pełny sweep skal w K4), zaszumiony `0.966512 ± 0.006572` (3 seedy) → `degradation_factor = 3.236`. NSF (`trained_with_noise_augmentation = true`, wartość zaszyta w `a03:123`, potwierdzona lekturą `preprocessing.py:146-149`, `train.py:161` i in.): czysty `0.384621` (ramię **publikowane**, `median`), zaszumiony `0.556800 ± 0.000611` → `degradation_factor = 1.448`. Porównanie na ramieniu publikowanym: `claimed_ratio = 1.845`, `ratio_clean = 1.288`, `ratio_noised = 0.576` (kierunek odwrócony). Liczby NSF niepowtarzalne w 3. cyfrze znaczącej (§1, niezaseedowany `model.sample()`) — cytować z podaniem przebiegu, tu `20260729`. Wspierające: `a09`, `a05` (§K5) |
| co by to obaliło | `sign_flips = false`, czyli znak `(ExoBiome − baseline)` niezmienny wobec konwencji wejścia. Osobno upada część o asymetrii, gdyby `holdout_metrics.json` baseline'u powstał na widmie czystym (`sample_noise=False`), jedna linia do sprawdzenia, `train.py:267`. Albo gdyby w ścieżce ExoBiome znalazła się jakakolwiek augmentacja szumem (skan `models/ariel_exobiome/*.py` dziś jej nie znajduje, §0a). Czego **nie** obala: sama wielkość ilorazu 1,29× ani kierunek na ramieniu `mean` — ustalenie dotyczy **zmiany znaku** między konwencjami, nie konkretnej wartości |
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

Zacommitowanych `holdout_metrics.json = 0.299376` i `validation_metrics.json = 0.293614` **nie odtwarza
żadna** z przemiatanych skal (`scales_reproducing_the_published_number = []`). Najbardziej prawdopodobne
wyjaśnienie — ta sama ewaluacja przy skali 1.0, wykonana na CUDA + AMP — pozostaje **hipotezą**. Skala 1.0
odtwarza wyłącznie re-ewaluację na Macu (`0.298693`), a sam check odnotowuje (`a04:34-36`), że trafienie
w liczbę z Maca niczego o liczbie opublikowanej nie ustala. Prowenienacja zacommitowanych liczb jest więc
**nieustalona**, nie ustalona jako CUDA + AMP.

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

Rozróżnienie: **model nigdy nie był trenowany przy skali
1.0**, a mimo to flagowe metryki policzono, ustawiając mnożnik na 1.0 przy ewaluacji. Zarzut „liczba powstała w punkcie pracy, w którym model nie był ani trenowany, ani
walidowany", przy czym punkt walidowany (0,5) daje wynik **lepszy** (0.295552 wobec 0.298693). To zamyka
warunek `max(quantum_scale) ≥ 1.0` z tabeli K5 rekordem treningu, a nie inferencją ze sweepu.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a04_quantum_scale_provenance.py` |
| komenda | `./.venv-qml/bin/python audit/a04_quantum_scale_provenance.py` |
| kryterium PASS | skala odtwarzająca opublikowane liczby == skala z chwili selekcji odczytana z `history.csv`. Tolerancja dopasowania: `--tol = 2e-5` (`a04:45`) — wartość **przyjęta, nie wyprowadzona**, i jedyny próg w tym checku. Dlatego rozstrzygające jest sformułowanie **niezależne od progu**: najbliższa skala jest **inna na każdym splicie** — na holdoucie 1.0 (`\|diff\| = 6,8e-4`), na validation 0.25 (`9,6e-4`). Żadna pojedyncza skala nie jest najbliższa obu zacommitowanym liczbom jednocześnie, przy **dowolnej** tolerancji. Przy progu rzędu szumu CUDA+AMP (~1e-3) zmieniłby się tylko powód, dla którego prowenienacja jest nieustalona, nie sam wniosek |
| wartość oczekiwana | `status = FAIL`; `best_epoch = 6`, `selection_time_quantum_scale = 0.5`, `selection_time_val_mrmse = 0.290811`; `scales_reproducing_the_published_number = []` i `published_number_reproduced_by_any_swept_scale = false` na **obu** splitach (holdout `n_rows = 4143`, validation `n_rows = 4142`) — **żadna** przemiatana skala nie odtwarza liczby zacommitowanej; odtwarzana jest wyłącznie re-ewaluacja na Macu, przy skali 1.0 (`scales_reproducing_only_the_mac_reevaluation_not_published = ["1.000000"]`, `matches_mac_reeval = true` tylko tam); `gate_off_mrmse = 0.302409` (holdout) i `0.299271` (validation); `quantum_pathway_contribution_at_scale_1 = 0.003716` (holdout) / `0.004450` (validation); `..._at_selection_scale = 0.006857` (holdout) / `0.007034` (validation); `best_scale_on_this_split = 0.666667`, `reported_is_best = false`. Uwaga o artefakcie: `training_state.json` zawiera **własne** pole `quantum_scale = 0.6667` (stan na moment przerwania runu), konkurencyjne wobec odczytu z `history.csv` w epoce 6 (`0.5`), którego `a04` używa zgodnie z zadeklarowanym kryterium (`a04:50-54`). Skala selekcji to `0.5`; `0.6667` z `training_state.json` opisuje koniec runu, nie epokę 6 |
| co by to obaliło | gdyby `training.py:771-772` przekazywało `quantum_scale=quantum_scale` (jak linia 565-572), albo gdyby sweep przy 0.5 dawał `matches_reported`/`matches_mac_reeval = true`. Osobno: `reported_is_best = true` (albo `best_scale_on_this_split = 1.000000`) usuwa część (b) zarzutu, bo raportowany punkt pracy byłby optimum sweepu, choć nadal niewalidowanym; części (a) — braku treningu przy skali 1.0 — to nie dotyczy |
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

**Zarzut rozkłada się na trzy osobne ustalenia — i pierwsze z nich jest na korzyść projektu.**

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
3. **Wzrost na epoce 7 jest zbieżny z odmrożeniem — ale przyczyny nie da się tu rozdzielić.** Skok
   val 0,29081 → 0,32866 (+13,0 %) pada dokładnie na przejście `backbone_frozen 1 → 0`. W tej samej
   epoce zmienia się jednak **także** `quantum_scale` (0,500 → 0,583), a K4 pokazuje, że skala wpływa
   na wyjście. Przy **jednym** przejściu obie zmiany są nierozłączne, więc odmrożenie jest hipotezą
   najbardziej prawdopodobną, nie ustalonym skutkiem. Rozdzieliłby je jeden przebieg z zamrożonym
   backbone'em i tą samą rampą skali.

**Stop był ręczny:** `best_epoch = 6`, `current_epoch = 8`, `early_stop_patience = 8`
(`training_state.json`), czyli wykorzystane **2 z 8** epok cierpliwości. Harmonogram runu go nie zatrzymał.

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a05_training_completeness.py` |
| komenda | `./.venv-qml/bin/python audit/a05_training_completeness.py` |
| kryterium PASS | run zakończony własną regułą ∧ raportowana metryka na plateau ∧ metryka selekcji == metryka raportowana ∧ selekcja na pełnym splicie |
| wartość oczekiwana | `status = FAIL`, `n_runs_with_issues = 4`. Rozstrzygające dla `exobiome_stage2_v4`: `epochs_run = 8` przy `max_epochs = 30` i `early_stop_patience = 8`, `best_epoch = 6`, `backbone_frozen_trajectory = [1,1,1,1,1,1,0,0]`, `val_trajectory` monotonicznie malejąca przez sześć zamrożonych epok (`0.293325 → 0.290811`), z przeskokiem `0.290811 → 0.328656` między epoką 6 i 7, `max(quantum_scale_trajectory) = 0.666667`, `selection_metric = "val mRMSE (full split, every epoch)"` ≠ `reported_metric = "val/holdout mRMSE at quantum_scale=1.0"`.|
| co by to obaliło | `epochs_run == max_epochs` (czyli 30) albo `epochs_run − best_epoch ≥ early_stop_patience` (≥ 8) - run zakończył się legalnym early stopem, nie ręcznie. Dalej: `best_epoch > quantum_backbone_freeze_epochs`, teza „model nigdy nie trenował z odmrożonym backbonem" upada; `val[best] ≤ val[best−1]` - słowo „dywergencja" jest nadinterpretacją; `max(quantum_scale) ≥ 1.0` → zarzut „wersji przy skali 1.0 nikt nie wytrenował" upada; pojawienie się checkpointu stage-1 w repo - budżet treningowy staje się udokumentowany |
| rekord | `reports/audit/20260729/a05_training_completeness.json` |

Ustalenie ma **drugą ścieżkę odtworzenia**, niezależną od `audit_lib`: one-liner czytający wyłącznie
surowe artefakty leży w `audit/README.md`, sekcja „Second reproduction paths” (K5). Weryfikuje samo
ustalenie, a nie kod audytu; wartości oczekiwane są w tabeli wyżej.

Oczekiwane wyjście: T1 ostatnia epoka 8 < max 30, od najlepszej 2 < patience 8, czyli **stop ręczny** ·
T2 `best_epoch = 6 == freeze_epochs = 6` · T3 `0,29081 → 0,32866`, **+13,0 %** · T4 max skali
**0,6667**. Warunki obalające każdy z tych testów są w wierszu „co by to obaliło" wyżej.

### K6. „Wkład kwantowy" nieodseparowalny od parametrów klasycznych

**Gdzie znaleziony.** `artifacts/ariel_quantum_best_v4_epoch6/best_model.pt` (`model_state_dict`) w zestawieniu
z `models/ariel_exobiome/model.py` — moduły `projector`, `quantum_block`, `quantum_head`, `quantum_gate`
składają się na „ścieżkę kwantową".

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
  **+36,68 %** względem modelu classical-only (189 254). Ablacja on/off **nie jest**
  param-matched.

„Korekta kwantowa" jest drugą pełną głową klasyczną z 8 dodatkowymi cechami. W rozbiciu modułów: `quantum_head = 51 845` wobec `classical_head = 50 309`. Jest to po prostu druga, większa głowa
klasyczna. Test ON/OFF mierzy „druga głowa włączona/wyłączona", nie „kwant włączony/wyłączony". Stąd wniosek,
który **nie zależy od żadnego eksperymentu**: przy tej architekturze wynik zerowy jest **nieinformatywny**
(obwód nie miał jak zadziałać: jego wyjście zajmuje **8 z 264** wymiarów wejścia głowy i jest mnożone przez
`|tanh(gate)|·scale ∈ [0.0178, 0.0292]` — obie liczby są w payloadzie `a06`), a wynik dodatni
**nieatrybuowalny** (**+36,68 %** dodatkowej pojemności klasycznej jest
wystarczającym wyjaśnieniem alternatywnym).

**Co go weryfikuje.**

| | |
|---|---|
| check | `audit/a06_param_accounting.py` |
| komenda | `./.venv-qml/bin/python audit/a06_param_accounting.py` |
| kryterium PASS | gałąź kwantowa dokłada < 1 % dodatkowych parametrów klasycznych względem modelu classical-only |
| wartość oczekiwana | `status = FAIL`; `total_params = 258688`, `quantum_pathway_params = 69434`, `circuit_params = 24`, `extra_classical_params_added_by_the_branch = 69410`, `classical_only_params = 189254`; `circuit_share_of_model = 9.277585e-05` (0,009278 %), `extra_classical_over_classical_only = 0.366756` (+36,68 %); `quantum_head_input_dim = 264`, `quantum_features_share_of_head_input = 0.030303` (3,03 %); `variational_layers = 1`, `circuit_formula = "3 * n_qubits * (depth // 2)"`; `effective_multiplier_at_scale_0.5 = [0.017765, 0.029234]` |
| co by to obaliło | `extra_classical_over_classical_only < 1 %` — wtedy ablacja on/off byłaby param-matched; `circuit_params ≠ 24` lub kształt `quantum_block.weights` niezgodny z `3·8·(2//2)`; albo `quantum_head_input_dim ≈ n_qubits` zamiast 264 — wtedy głowa nie byłaby full-context i zarzut „druga głowa klasyczna" słabnie |
| rekord | `reports/audit/20260729/a06_param_accounting.json` |

Ustalenie ma **drugą ścieżkę odtworzenia**, niezależną od `audit_lib`: one-liner czytający wyłącznie
surowe artefakty leży w `audit/README.md`, sekcja „Second reproduction paths” (K6). Weryfikuje samo
ustalenie, a nie kod audytu.

Oczekiwane wyjście: T1 `3·8·(2//2) = 24`, kształt `(24,)` · T2 udział obwodu **0,009278 %** ·
T3 ścieżka 69 434 / kwantowe 24 / **klasyczne 69 410** · T4 classical-only 189 254, branch **+36,68 %** ·
T5 wejście głowy **8 z 264 = 3,03 %**.

### K7. Model sam wyciszył kwant: twierdzenie niezgodne z kodem.

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
| co by to obaliło | init inny niż `torch.zeros` (np. `randn`/`ones`) `sign_agreement_across_gases = true` → `mean\|tanh\|` byłoby adekwatnym podsumowaniem i ta część zarzutu słabnie. `converged_by_own_rule = true` wtedy 0,046 jest stanem równowagi. Mnożnik `\|tanh(gate)\|·scale` rzędu 1 zamiast ~0,02 — wtedy korekta kwantowa miałaby wpływ porównywalny z głową klasyczną.|
| rekord | `reports/audit/20260729/a07_gate_dynamics.json` |

Ustalenie ma **drugą ścieżkę odtworzenia**, niezależną od `audit_lib`: one-liner czytający wyłącznie
surowe artefakty leży w `audit/README.md`, sekcja „Second reproduction paths” (K7). Weryfikuje samo
ustalenie, a nie kod audytu.

Oczekiwane wyjście: `grep` pokazuje `nn.Parameter(` + `torch.zeros(len(TARGET_COLUMNS), …)` łamane
na trzy linie (`model.py:260-262`) · T1 `mean|tanh| = 0,046293` · T2 zgodne znaki **False** (CO₂ `+`,
cztery pozostałe `−`) · T3 mnożnik `|tanh(gate)|·0,5 ∈ [0,0178; 0,0292]` · T4 ~12 426 kroków przy
`lr = 2e-4`.

---

### K9. Niepełny wektor celu: brak temperatury i promienia.

**Gdzie znaleziony.** Głównym dowodem pomiarowym jest `audit/a15_target_completeness.py` — liczy
sprzężenia z `Tracedata.hdf5`, wyciek T przez aux na ADC, wartość kondycjonowania na prawdziwym
`(T, R_p)` oraz część cross-generator. Definicje celu i hardkody cech, na których te pomiary stoją,
są w kodzie modeli.

**1. Pięć vs siedem celów.** `models/ariel_exobiome/constants.py:29-35` — `TARGET_COLUMNS` ma
dokładnie pięć pozycji (`log_H2O`, `log_CO2`, `log_CO`, `log_CH4`, `log_NH3`). Cel benchmarku ADC2023
to **7 parametrów** (`planet_radius`, `planet_temp` + 5 gazów) — tyle ma `FM_Parameter_Table.csv`,
`QuartilesTable.csv` i każdy trace w `Tracedata.hdf5`. `a15` porównuje oba wektory wprost:
`TRACE_COLS = ["planet_radius", "planet_temp", *A.TARGETS]` → `n_predicted = 5`, `n_required = 7`.

**2–4. `temperature_k` nie dociera do żadnego modelu crossgen.** Wszędzie
ten sam mechanizm: aux budowane jest ze stałych, a `temperature_k` albo nie jest czytane w ogóle,
albo tylko walidowane jako kolumna etykiet. `a15` przeszukuje cztery pakiety i potwierdza brak
referencji jako cecha wejściowa (`crossgen.temperature_k_referenced_in_package`).

| pakiet | gdzie budowane jest aux | co z `temperature_k` |
|---|---|---|
| `taurex_exobiome` | `models/taurex_exobiome/dataset.py:315-335` — `star_temperature`, `planet_distance` z `TAUREX_FIXED_*` | nie czytane |
| `taurex_exobiome_without_quant` | `models/taurex_exobiome_without_quant/dataset.py:327-347` — bliźniaczo | nie czytane |
| `taurex_fmpe` | `models/taurex_fmpe/raw_dataset.py:94-109` z hardkodu; `AUX_FEATURE_COLS` (`models/taurex_fmpe/constants.py:24-33`) bez `temperature_k` | nie czytane |
| `ariel_winner_on_taurex` | `models/ariel_winner_on_taurex/prepare_dataset.py:113-130` z `FIXED_STAR_TEMPERATURE_K` / `FIXED_PLANET_DISTANCE_AU` | tylko w `REQUIRED_LABEL_COLUMNS` (`models/ariel_winner_on_taurex/prepare_dataset.py:40-48`, `temperature_k` w `:46`), nie jako cecha |

Reprezentatywny cytat, `models/taurex_exobiome/dataset.py:320-331` — wzorzec identyczny we wszystkich
czterech:

```python
star_mass_kg   = np.full(row_count, TAUREX_FIXED_STAR_MASS_KG, dtype=np.float32)
planet_distance = np.full(row_count, TAUREX_FIXED_PLANET_DISTANCE_AU, dtype=np.float32)
...
"star_temperature": np.full(row_count, TAUREX_FIXED_STAR_TEMPERATURE_K, dtype=np.float32),
```

Konsumpcja jest po stronie `transform_aux_features` (`models/taurex_exobiome/dataset.py:366-371`),
które bierze wyłącznie `frame[AUX_COLUMNS]`, a `AUX_COLUMNS`
(`models/ariel_exobiome/constants.py:8-17`) to osiem kolumn gwiazdowo-orbitalnych —
`star_temperature` i `planet_distance` tam są, `temperature_k` nie ma.

**Po co w retrievalu w ogóle jest temperatura czyli trochę teorii.**

Line & Parmentier 2016 (arXiv:1511.09443, eq. 5) i HK17 (arXiv:1702.02051, eq. 12) rozdzielają dwie rzeczy:

| co widmo mierzy | od czego zależy w λ | rola `T` |
|---|---|---|
| **amplituda / „wysokość" cech** | wspólny mnożnik `H = kT/(μg)` na całe widmo | **główna** — `T` wchodzi liniowo w `H` |
| **kształt** (gdzie są pasma, stromość zboczy) | `d(ln σ_λ)/dλ` — fizyka linii; mieszanka `κ(λ) = Σ X_i σ_i(λ)` | **drugi rząd** — `σ(T)`, chmury, `μ` zależne od składu, profil T–P |

W eq. 5 abundancja **nie występuje** w pochodnej nachylenia stąd sprzężenie T↔abundancja mierzone
w (a)–(b) idzie przede wszystkim przez **skalę wysokości i normalizację** (`P_ref·X`, `R_ref`), a nie
przez to, że `T` przesuwa linie w λ. Efekty drugiego rzędu mogą dodatkowo zmieniać kształt, ale to
nie jest główny mechanizm liczony w tabelach poniżej.

**ExoBiome nie przewiduje** `planet_temp` ani `planet_radius`. Na ADC `T` jest w praktyce
**wyciekana przez aux** (`r ≈ 0,99`, patrz (c)); na crossgen (tau) pozostaje **niekontrolowanym
parametrem** skalującym amplitudę wszystkich cech (patrz (d)). Pomiary (a)–(e) dotyczą **posteriora
ADC** (`Tracedata.hdf5`) oraz **generatora tau** w crossgen — **nie** `poseidon_test` (K1).

(a) Sprzężenie w posteriorze referencyjnym — ważone korelacje w trace'ach nested sampling, mediana po
663 planetach holdoutu — czyli na wszystkich, które mają 7-kolumnową macierz trace (**663 z 4143, 16 %**).

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
2. **R_p ↔ T to najsilniejsza korelacja w całej macierzy 7×7**. Model nie
   ma więc żadnego uchwytu na dominującą degeneracją problemu.

(b) Kurs wymiany T → dex. ważone nachylenie MNK $\log X$ na $T$ w próbkach posterior w posteriorze,
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
podanie *prawdziwych* (T, R_p) poprawia gazy tylko o **0,3 %** (mRMSE 0.4354 → 0.4343) — nie ma czego
dodać, bo T już tam jest. Argument „brak T psuje dokładność" na ADC **nie broni się**.

I zostaje wniosek o samym benchmarku: **retrieval referencyjny nigdy nie widział tabeli aux.** Jego
własny posterior dla T ma medianę σ = 38.4 K przy RMSE mediany wobec prawdy **201.3 K** (przeufny
**5,2×**), a na promieniu jest przeufny **16,3×** (σ 0.000967 wobec RMSE 0.01578,
`reference_retrieval_precision.planet_radius.overconfidence_ratio`) — podczas gdy aux daje T z błędem
48.5 K, czyli **4× lepiej niż dokładny retrieval bayesowski**.

(d) Na zbiorze cross-generator temperatura jest nieobecna całkowicie — i to jest realny problem
fizyczny. `temperature_k` przebiega 500–1800 K (**3,6× w skali wysokości**), a
`_build_taurex_auxiliary_frame` hardkoduje `star_temperature = 5500 K` i `planet_distance = 0.05 AU`,
więc trasa przez temperaturę równowagową znika: `corr(T_eq_from_aux, temperature_k) = +0.0003`.
Jednocześnie `temperature_k` **nie jest czytane** przez `taurex_exobiome`,
`taurex_exobiome_without_quant` ani `taurex_fmpe` (w `ariel_winner_on_taurex` występuje wyłącznie na
liście `REQUIRED_LABEL_COLUMNS` — `models/ariel_winner_on_taurex/prepare_dataset.py:46` — też nie jest
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
| wartość oczekiwana | `n_predicted = 5`, `n_required = 7`, status FAIL, `status_terms.decisive = "coupling"`. Cztery liczby rozstrzygające, wszystkie na **663 planetach / 3315 parach** (cały dostępny materiał): iloraz sprzężeń T↔gazy do gaz↔gaz `4.1385` (`0.22123` / `0.05345`), `med_abs_r_radius_temperature = 0.77660`, mediana kursu `0.352` dex/100 K (p90 `1.000`), wyciek T przez aux `r = 0.99276` przy RMSE `48.496 K` i `sd(T) = 401.385 K`. Reszta bloków — per-gaz sprzężenia i nachylenia, ogony p10, kondycjonowanie `0.00259`, odtwarzalność ADC (R² `0.99010` / `0.97955`, RMSE `0.04539` / `57.406 K`), crossgen (`corr = +0.00031`, R² z widma `0.87218`, promień `0.31807`) — cytowana w (a)–(e) wyżej i w rekordzie |
| co by to obaliło | **Człon rozstrzygający** (`status_terms.decisive = "coupling"`) upada, gdy **jednocześnie** `mean_med_abs_r_temperature_gas / mean_med_abs_r_gas_gas` ≤ **1,5** oraz `med_abs_r_radius_temperature` ≤ **0,30** — te dwa progi są zaimplementowane w `a15:393-394`, a **trzeci** — `r² ≥ 0,95` dla `temperature_k` i `planet_radius_rjup` na crossgenie, czyli operacyjna definicja czlonu „wyznaczone przez wejście” — w `a15:395`. **Żadnego z tych trzech** nie ma ani w polu `pass_criterion` payloadu, ani w wierszu „kryterium PASS" wyżej; to luka w samym harnessie, nie w ustaleniu. Żaden z nich nie jest wyprowadzony. Dziś zmierzono `4,1385` i `0,7766` (oba z dużym zapasem powyżej progów) oraz `r² = 0,87218` i `0,31807`, czyli oba pod 0,95 — czlon „wyznaczone przez wejście” też nie zachodzi. **Osobno** upadają punkty (iii) i (iv), gdyby `temperature_k` było podawane modelom na crossgenie jako cecha albo gdyby model miał 7 wyjść — ale to nie usuwa członu rozstrzygającego |
| rekord | `reports/audit/20260729/a15_target_completeness.json` |


### K9(f). Co przewidują modele w literaturze — i dlaczego 5 gazów nie ma precedensu

Ustalenie dotyczy **literatury**, nie tego repo, więc nie ma i nie może mieć checku.

Przegląd ML/SBI dla retrievalu atmosfer (2016–2026): domyślny wektor celu to „temperatura +
abundancje + kotwica rozmiaru/ciśnienia", a liczby celów grupują się w dwóch reżimach — 5–10 dla
transmisji (Márquez-Neila arXiv:1806.03944, Cobb arXiv:1905.10659, ExoGAN arXiv:1806.02906,
Yip arXiv:2011.11284, Ardévol Martínez arXiv:2203.01236, Exoformer arXiv:2603.27623, ADC) i 16 dla
emisji w petitRADTRANS (Vasist arXiv:2301.06575, Gebhard arXiv:2312.08295), gdzie nadwyżką jest
mikrofizyka chmur i profil T–P. Degeneracja ze skalą
wysokości adresowana zawsze jawnie: dopasuj T i zamroź R₀/P₀, albo dopasuj T i P_ref
(Nixon & Madhusudhan arXiv:2004.10755), albo dopasuj T i R_p (ADC2023, ExoGAN, Yip, Exoformer).

Tylko w dwóch pracach nie jest to uzywane, jeden raz uzasadnione. *INARA* (arXiv:1811.03390; Zorzan 2025) ma
`FC(12)` i same abundancje, przy czym promień, masa i profil P–T **są losowane** przy generowaniu 3 mln
widm i nigdy nieprzewidywane, czyli działają jako niemodelowana wariancja zakłócająca. Praca tego nie
uzasadnia, a autorzy sami nazywają komponent ML dowodem koncepcji (następca, arXiv:2508.00076, dodaje
10 celów). Precedens na *uzasadnione* pominięcie daje tylko Exoformer: usuwa masę planety z powodu
degeneracji — i **zamraża ją**, zamiast ignorować.

**Wniosek.** Zawężenie do 5 gazów jest bez precedensu w klasie, do której ExoBiome należy (swobodna
chemia, T losowana niezależnie). Dopuszczalne ścieżki: przewidywać 7 parametrów, albo jawnie
kondycjonować na T i R_p i tak to nazwać.

---

### K10. Raportowana metryka nie jest metryką benchmarku

**Gdzie znaleziony.** Oficjalny kod punktujący leży w repo: `models/adc_baseline/posterior_utils.py:84`
(`score_trace.append((1 - metric_ks.statistic) * 1000)`) oraz `models/adc_baseline/spectral_metric.py:43`
(`score = 1000-np.mean([bound_loss,median_loss])`; sygnatura funkcji w `:19`). Liczby raportowane jako wynik to mRMSE z
`reports/model_comparison/rmse/exobiome_metrics.json`, `sota_metrics.json` i `cnn_metrics.json`.

**Wyjaśnienie.** Oficjalna punktacja ADC2023 to

> `score = 0.8 × score_posterior + 0.2 × score_spectral`

gdzie `score_posterior` to **dwupróbkowy test Kołmogorowa–Smirnowa na 7 rozkładach marginalnych**
wobec próbek referencyjnych MultiNest, a `score_spectral` to odwrócona strata Hubera na medianie
widma i IQR. Potwierdzone niezależnie przez trzy prace uczestników (Aubin arXiv:2309.09337; Unlu i in.
arXiv:2310.10521; Sweet arXiv:2406.10771). **mRMSE nie jest metryką tego challenge'u w żadnym stopniu.**

Konsekwencje, każda osobno poważna:

1. **„ExoBiome bije zwycięzcę ADC2023" jest zdaniem o metryce, której challenge nie używał**, i której
   zwycięzca nie optymalizował. Zestawione z K3 (różne wejścia) to
   **trzeci niezależny powód**, dla którego to porównanie jest nieważne.
2. **Model punktowy jest na metryce KS niepunktowalny.** KS wymaga rozkładu; ExoBiome go nie
   produkuje. Nie da się więc podać liczby, która stawiałaby ExoBiome na tej samej skali co
   zgłoszenia challenge'owe. Zastrzeżenie, którego nie wolno pominąć: to zdanie dotyczy **KS**, czyli
   członu o wadze 0,8, a **nie** wszystkich metryk rozkładowych. `a24.skill_vs_prior` daje
   `ks.exobiome = −0,2305` (ujemny — i to on decyduje o statusie), ale **`w1_dex.exobiome = +0,4459`**
   i **`light_track.exobiome = +0,5384`**, czyli na dwóch z trzech mierzonych wielkości rozkładowych
   estymator punktowy **bije ramię PRIOR**. Generalizacja „każda metryka rozkładowa stawia model przy
   najgorszej możliwej wartości" jest więc **za mocna** — również w polu `interpretation` samego `a24`.
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
| co by to obaliło | (1) oficjalny dokument organizatorów, który podawałby mRMSE jako metrykę ADC2023, albo **6** celów zamiast 7 — strona challenge'u w `/documentation/data` podaje wciąż 6 (nieaktualna treść ADC2022) i **nie ma oficjalnego papieru ADC2023 na arXiv ani w czasopiśmie**. Dopóki to nie jest domknięte, specyfikacja stoi na trzech pracach uczestników plus na kodzie zawendorowanym w `models/adc_baseline/`, nie na dokumencie organizatorów; (2) przeformułowanie claimu z „bijemy zwycięzcę ADC2023" na „punktowa dokładność na poziomie SOTA przy N× niższym koszcie" (wariant, który `docs/VERIFICATION.md:128-130` sam dopuszcza) — wtedy zdanie przestaje być zdaniem o metryce challenge'u i punkt 1 przestaje mieć adresata; (3) punkt 2 upada dopiero po dodaniu głowy rozkładowej **i** siedmiu wyjść jednocześnie — samo jedno z dwóch nie wystarcza |
| rekord | `reports/audit/20260729/a24_official_metrics.json` |

---

### K11. Nasza implementacja TauREx też ma wadę fizyczną: brak absorpcji indukowanej zderzeniami (CIA)

**Gdzie znaleziony.** `data/crossgen_biosignatures/taurex_backend.py:154-157` — lista `contributions`
przekazana do `TransmissionModel` (`:148`) ma dwa elementy i nie ma wśród nich `CIAContribution`.
To jest **jedyne** miejsce, w którym generator tau konfiguruje wkłady do opacity: w całym pliku jest
jedna konstrukcja `TransmissionModel` (`:148`), jedna lista `contributions=` (`:154`) i **zero** wywołań
`add_contribution`, więc defekt ma dokładnie jeden punkt wejścia i jedno miejsce naprawy.
Zdania, które to podważa: wszystkie liczby cross-generator raportowane jako **bezwzględne** abundancje
oraz nazwa „cross-generator gap" w `reports/`.
**Wyjaśnienie.** `data/crossgen_biosignatures/taurex_backend.py:154-157`:

```python
            contributions=[
                imports["AbsorptionContribution"](),
                imports["RayleighContribution"](),
            ],
```

(klasy wchodzą przez słownik `imports`, wypełniony w `:74-75` z `taurex.contributions`; semantyka jest
ta sama co przy bezpośrednim `AbsorptionContribution()`.) Brakuje `CIAContribution`. W atmosferze zdominowanej przez H₂/He absorpcja indukowana zderzeniami
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

**POSEIDON w tej konfiguracji włącza CIA domyślnie — asymetria jest jednostronna.**
`poseidon_backend.py:103-114` woła `define_model(...)` bez żadnego przełącznika CIA
(`bulk_species=["H2","He"]`, ślady = 5 gazów, brak argumentu `disable_continuum`); w `define_model`
CIA **nie jest opcją wywołania**, bo biblioteka buduje `CIA_pairs` sama, filtrując `supported_cia`
po `chemical_species = bulk_species + param_species`.

*Dowód na wersji generującej (1.3.2).* Atmosfera zbudowana przez `make_atmosphere` **na przypiętym
1.3.2** — tej, która wygenerowała zbiór — zawiera tablicę mieszanin CIA: w
`reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json` pole
`payload.stage1.grids.repo_increasing.atmosphere_keys` obejmuje **`X_CIA`** obok `X`, `X_active`,
`X_bf`, `X_ff`. CIA jest po stronie POSEIDON-a częścią konstrukcji modelu, nie opcją.

NOTE: w odróżnieniu od K1 dane mają strukturę i model się na nich
uczy (skill 0,50 na tau/val). Ale każde twierdzenie o **bezwzględnych** abundancjach z tego zbioru
dziedziczy przesunięcie rzędu dex.

Dodatkowe dowody na kodzie **spoza** audytu.

*Kotwica A — oryginalny kod organizatorów, w repo, niezależny od wersji.*
`models/adc_baseline/FM_utils_final.py:167-211` to **oficjalny forward model ADC2023** (docstring:
„Initialise the official forward model for ADC2023"), czyli ta sama biblioteka i to samo pasmo, na
których powstał zbiór ADC. Konfiguruje trzy wkłady jawnie: `add_contribution(AbsorptionContribution())`,
`RayleighContribution()`, **`CIAContribution()`** (`:207-209`), i osobno ładuje bazę:
`CIACache().set_cia_path(CIA_path)` (`:183`). Gdyby CIA wchodziło implicite, organizatorzy nie
dodawaliby go trzecim wywołaniem.

*Kotwica B — źródło TauREx3, wywołanie odtworzone.* Lista wkładów startuje
pusta (`taurex/model/model.py:42`: `self.contribution_list = []`) i jedyną drogą wejścia jest
`add_contribution` (`model/model.py:57-65`), wołane albo jawnie, albo z pętli po `contributions=`
(`model/simplemodel.py:121-123`). Żadna gałąź nie dokłada wkładu implicite. Odtworzone oba wywołania:

```
NASZE      (taurex_backend.py:154-157)   → ['Absorption', 'Rayleigh']
OFICJALNE  (FM_utils_final.py:207-209)   → ['Absorption', 'Rayleigh', 'CIA']
```

Czyli brak `CIAContribution` w liście **oznacza brak CIA w widmie**. Nie jest to wada generatora, wynika ona w pełni ze złej realizacji w naszym kodzie.

**Co go weryfikuje.**

| | |
|---|---|
| check | **brak checku**, weryfikacja przez **lekturę trzech źródeł spoza audytu**: `data/crossgen_biosignatures/taurex_backend.py:148,154-157`, `models/adc_baseline/FM_utils_final.py:183,207-209` (oficjalny FM ADC2023 organizatorów — trzy wkłady + `CIACache`), oraz `taurex/model/model.py:42,57-65` + `simplemodel.py:121-123` w zainstalowanym `taurex 3.3.2` (brak domyślnego CIA). Po stronie POSEIDON-a dowód jest na wersji generującej: `X_CIA` w `atmosphere_keys` rekordu **1.3.2** (`reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json`) |
| wartość oczekiwana | Z lektury kodu: `contributions` ma **2** elementy (`AbsorptionContribution`, `RayleighContribution`), zero wystąpień `CIAContribution` w `data/crossgen_biosignatures/`; zbiór tau **41 423** widma (`tau/train 37 281` + `tau/val 4 142`), pasmo 0,6–5,25 µm (`manifest.json: wavelength_grid.min_um = 0.6`, `max_um_actual_edge = 5.2506`; `max_um_requested = 5.2`), 218 binów, `PRESSURE_LEVELS = 100`, `PRESSURE_MIN_BAR = 1.0e-6`, `PRESSURE_MAX_BAR = 1.0e2`; tło H₂/He **zrealizowane**: `vmr_h2 ∈ [0.8275, 0.8500]`, `vmr_he ∈ [0.1460, 0.1500]` (`a21`) — niższe od deklarowanych 0,85/0,15, bo gazy śladowe zabierają do `TRACE_VMR_MAX_TOTAL = 0.10`.|
| co by to obaliło | **(1)** Wystąpienie `CIAContribution` w liście `contributions` w `data/crossgen_biosignatures/` — wtedy generator jednak liczy kontinuum i ustalenie upada w całości. **(2)** Wykazanie, że `taurex` dokłada CIA **implicite**, mimo pustej listy. `taurex/model/model.py:42`, `:57-65` i `simplemodel.py:121-123` tego nie potwierdza, ale **to lektura na 3.3.2, nie pomiar na 3.2.4**: instalacja 3.2.4 i powtórzenie tej lektury jest najtańszym sposobem domknięcia tej luki. **(3)** Odtworzenie zapisanego widma tau **bez** CIA z niezerową rozbieżnością. **(4)** Po stronie POSEIDON-a: brak `X_CIA` w atmosferze zbudowanej na 1.3.2 albo wykazanie, że `CIA_pairs` w 1.3.2 wychodzi puste dla naszych `chemical_species`. |
| rekord | `—` |

---

## Załącznik A. Liczby kluczowe w jednym miejscu

| wielkość | wartość | źródło |
|---|---:|---|
| ExoBiome params / winner params | 258 688 / 10 771 200 (41,6×) | `a06`, `a03` |
| obwód / model | 24 / 258 688 = 0,0093 % | `a06` |
| ścieżka kwantowa vs classical-only | +69 410 par. = +36,7 % | `a06` |
| `mean\|tanh(gate)\|` | 0.0463 (CO₂ przeciwny znak) | `a07` |
| ExoBiome holdout, kwant OFF | 0.302409 | `a04` |
| ExoBiome holdout, skala 0.5 (walidowana) | 0.295552 | `a04` |
| ExoBiome holdout, skala 0.6667 (optimum sweepu) | 0.295487 | `a04` |
| ExoBiome holdout, skala 1.0 — **opublikowana** (`holdout_metrics.json`, prowenienacja nieustalona, K4) | 0.299376 | `a04`, `a02` |
| ExoBiome holdout, skala 1.0 — re-ewaluacja na Macu (`mac_holdout_metrics.json`) | 0.298693 | `a04`, `a03` |
| wkład ścieżki kwantowej: @1.0 / @0.5 | 0.003716 / 0.006857 | `a04` |
| ExoBiome holdout, wejście +N(0,σ) | 0.966512 ± 0.006572 | `a03` |
| NSF holdout, wejście czyste, mean / median | 0.403603 / 0.384621 | `a03` |
| NSF holdout, wejście +N(0,σ), mean / median | 0.548927 / 0.556800 | `a03` |
| stosunek NSF/ExoBiome, ramię **median** (opublikowane): czyste / zaszumione | 1.288 / 0.576 (odwrócenie) | `a03` |
| baseline stały: ADC holdout / TauREx val / POSEIDON | 1.4404 / 2.8852 / 2.8940 | `a02` (średnia **pełnego** splitu treningowego) |
| skill: quantum / noquant / winner / H200 na POSEIDON | −0.111 / −0.133 / −0.193 / −0.000 | `a02` |
| drabina na ADC holdout: stała / tylko aux / tylko widmo (ridge, GBM) | 1.4405 / 1.4407 / 1.0174, 0.4409 | `a26` (stała z **prefiksu 12 000** wierszy — stąd 1.4405, nie 1.4404 jak w `a02`) |
| udział tabeli aux w skillu (ADC / tau-val) | **−0,19 % / −0,41 %** — aux nie dokłada nic ponad widmo | `a26` |
| ridge / GBM na POSEIDON (tylko-widmo) | skill **−0,019 / −0,127** — wspólny znak. MDE `a12` to **0,0436 dex** mRMSE, nie skill: po podzieleniu przez podłogę 2,8940 daje **0,0151 skillu**, więc wszystkie trzy wartości (0,019 / 0,127 / 0,111) leżą **nad** progiem. Zastrzeżenie: `sd = 0,409` dex zmierzono na parze quantum↔noquant, więc dla pary ridge↔stała jest pożyczone | `a26`, `a12` |
| referencyjny nested sampling, mRMSE / σ | 1.434 / 1.619 dex | `a08` |
| frakcja planet z posteriorem wielomodalnym | 31 – 62 % (per gaz, 200 planet, próba losowa `default_rng(42)`) | `a08` |
| statystyka szumu R: noiseless / 1.0σ / ADC | 0.370 / 1.093 / 0.757 | `a09` |
| MDE (80 %, α=0.05): n=685 / n=64 | **0.0439** / **0.1453** (estymand per-wiersz) | `a12 |
| MDE dla estymandu, który raporty faktycznie cytuja (agregat mRMSE), n=685 | **0.0436** | `a12` |
| liczba seedów w projekcie | 1 (42) | `a12` |
| liczby publikowane bez artefaktu, **wg `a13`** (brak wag ∧ metryk ∧ predykcji): `random_forest_holdout`, `winner_on_taurex_poseidon`, `h200_poseidon`, `garnet_hardware` | 4 | `a13` |
| liczby cytowane **bez pliku predykcji obok nich**, wg `a02`: CNN 0,6500, winner-on-TauREx 3,4531, H200 2,8946 | 3 | `a02` |
| ↳ dlaczego te listy się różnią: `a02` pyta, czy obok cytowanej liczby leży plik predykcji; `a13` pyta, czy claim ma **gdziekolwiek** wagi/metryki/kod. CNN 0,6500 jest dlatego w liście `a02`, ale **nie** w liście `a13` (tam `backed_summary`: wagi tak, metryki tak, predykcje NIE) | — | `a02`, `a13` |
| KS vs referencja: ExoBiome / NSF / **prior** / podłoga | 0.7508 / 0.4415 / **0.6101** / 0.0444 | `a24` |
| skill wobec priora na KS: ExoBiome / NSF | **−0.231** / +0.276 | `a24` |
| Wasserstein-1 [dex]: ExoBiome / NSF / prior / podłoga | 1.2293 / 1.0717 / **2.2186** / 0.0592 | `a24` |
| skill wobec prioru: KS / W1 / light-track (ExoBiome) | **−0,2305** / **+0,4459** / **+0,5384** — ujemny tylko na KS | `a24` |
| martwe importy | 17 w 7 plikach | `a14` |

---