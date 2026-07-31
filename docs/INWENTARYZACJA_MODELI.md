# Inwentaryzacja modeli — ExoBiome

Stan na przebieg `reports/audit/20260729/` (21 checków, `is_partial_run: False`, 0 ERROR).

**Zasada tego dokumentu:** każda komórka pochodzi z artefaktu albo z payloadu checku, nie z opisu
raportów zespołu. Gdzie liczba nie ma pokrycia w artefakcie, jest to napisane wprost. 
---

## 1. Tabela główna

| model | zbiór treningowy | zbiór oceniany | stan zbioru | wejście | parametry | stan treningu | uruchamialność |
|---|---|---|---|---|---|---|---|
| **ExoBiome hybrydowy** (`ariel_exobiome`) — model flagowy | ADC2023 train (33 138) | ADC2023 holdout (4143), validation (4142) | **zdrowy** | 52 biny `instrument_spectrum` + 52 `instrument_noise` + `instrument_width` + `wavelength_um` + 8 kolumn aux; normalizacja: dzielenie przez własną średnią | **258 688** ogółem; ścieżka kwantowa 69 434, z czego **24 kwantowe** (0,0093 % modelu) | 8 z 30 epok, **stop ręczny** (patience 8 niewyczerpane); best epoch 6 = ostatnia epoka z zamrożonym backbone'em; po odmrożeniu val +13,0 % | **pełna**, wszystko w `artifacts/ariel_quantum_best_v4_epoch6/`: wagi `best_model.pt`, predykcje `{holdout,validation,testdata}_predictions.csv`, metryki `{holdout,validation}_metrics.json`, `config.json`, `history.csv`, `scalers.json`, `split_manifest.json`. Rekonstrukcja pipeline'u zweryfikowana wiersz po wierszu wobec re-ewaluacji na CPU (`a27`, kolumna `mac_cpu`: 0 z 4143 wierszy powyżej tolerancji); wobec **opublikowanego** `holdout_predictions.csv` (CUDA+bf16) rozjazd ma 4143 z 4143 wierszy — patrz §1 raportu metodologicznego. Kod: `models/ariel_exobiome/` |
| **ExoBiome bez kwantów** (`taurex_exobiome_without_quant`) | crossgen tau train (37 281) | POSEIDON test (685), tau val (4142) | **POSEIDON zepsuty (K1), tau niepełny fizycznie (K11)** | jak wyżej, ale 218 binów `transit_depth_noisy` | suma parametrów **nie zmierzona** (`a06` liczy tylko wariant flagowy); zmierzona jest natomiast **gałąź rezydualna**: 85 733 par. (`a11.residual_branch_params.classical_control_arm`, z `named_parameters`) | `best_epoch` 59 przy `max_epochs` 60 — jedyny run, którego najlepsza epoka wypada na końcu harmonogramu (`a05` nie emituje dla niego `epochs_run`, więc liczby faktycznie przebiegniętych epok nie da się z rekordu odczytać) | **asymetryczna**, w `reports/taurex_noquant_taurex_snapshot_20260312_133054/`: strona **POSEIDON** ma predykcje `poseidon_{holdout,test}_predictions.csv` i metryki `poseidon_holdout_metrics.json` — mRMSE 3,279559 **odtworzone z predykcji** (przeliczone 3,2795596, Δ ≈ 6e-7 — metryka jest zapisana z sześcioma miejscami, więc zgodność bitowa nie jest tu definiowalna; `reports/taurex_model_comparison.md:23`, `a13` przy tolerancji 1e-5). Strona **tau/val** (1,423032) **nie ma pliku metryk ani predykcji** — istnieje wyłącznie jako proza w README (`taurex_model_comparison.md:23`). **Wag nie ma** (`best_model_epoch059.pt` brak — `a14`). Kod: `models/taurex_exobiome_without_quant/` |
| **ExoBiome kwantowy na TauREx** (`taurex_exobiome`) | crossgen tau train | POSEIDON test, tau val | **POSEIDON zepsuty (K1), tau niepełny (K11)** | jak wyżej | suma parametrów **nie zmierzona**; gałąź rezydualna (kwantowa) **69 434** par., z tego 24 kwantowe (`a11.residual_branch_params.quantum_arm`) | `best_epoch` 5 przy `max_epochs` 20; README własnego snapshotu stwierdza, że run **trwał dalej** → artefakt śródlotny | **asymetryczna**, w `reports/ariel_quantum_taurex_snapshot_20260312_1003/`: wagi `stage2_best_model_epoch005.pt`; strona **POSEIDON** — predykcje `poseidon_{holdout,test}_predictions.csv`, metryki `poseidon_holdout_metrics.json`, mRMSE 3,215615 **odtworzone bit-exact** (`taurex_model_comparison.md:22`). Strona **tau/val** (1,449002) **bez pliku metryk/predykcji**, „not yet re-derived" (tamże). Kod: `models/taurex_exobiome/` |
| **Reimplementacja NSF** (`adc_winner_on_ariel`) — model odniesienia | ADC2023 train (33 138) | ADC2023 holdout (4143) | **zdrowy** | 52 biny + szum, standaryzacja per widmo (mean 0, std 1) + mean/std jako 2 cechy + 4 estymatory promienia; **szum dolosowywany co epokę** | **10 771 200** (41,6× więcej niż ExoBiome) | 79 z 300 epok; stop na cierpliwości **val NLL**, podczas gdy metryka porównania (mRMSE) nadal spadała; LR zjechany 256× | **pełna**, w `models/adc_winner_on_ariel/trained_run/`: 3 checkpointy `best_model_by_mrmse.pt`, `best_model_by_nll.pt`, `resume_latest.pt`; metryki `{holdout,validation}_metrics.json`, `comparison_metrics.{csv,json}`; `settings_resolved.yaml`, `saved_split_manifest.json`, `train.log`. Kod: `models/adc_winner_on_ariel/` |
| **NSF na TauREx** (`ariel_winner_on_taurex`) | crossgen tau | POSEIDON test | **POSEIDON zepsuty (K1)** | 218 binów `transit_depth_noisy` + aux z hardkodu | nie zmierzone | brak danych o przebiegu | **brak**: liczba 3,4531 pochodzi z `reports/ariel_winner_on_taurex_20260312_112940_results_summary.md`, **bez artefaktu wag/predykcji obok niego**. Kod: `models/ariel_winner_on_taurex/` |
| **CNN — baseline organizatorów** (`adc_baseline`) | ADC2023, augmentacja szumem ×10 | ADC2023 holdout | **zdrowy** | 52 biny, MC-dropout aktywny w inferencji → wyjście rozkładowe | nie zmierzone | zawendorowany, gotowy | **inferencja możliwa**: wagi `models/adc_baseline/cnn_whole_ariel_new.weights.h5` i `cnn_cnn.weights.h5` (LFS) + kod punktujący w tym samym katalogu. Liczba 0,6500 cytowana w `reports/model_comparison/rmse/cnn_metrics.json`, **bez pliku predykcji obok niej** |
| **NSF rodziny zwycięskiej** (`ariel_winner_trace_nf`) | — | — | — | cele z `Tracedata.hdf5`, **7 parametrów** | — | **nigdy nie trenowany** — `best_independent_bundle.pt` nie istnieje w żadnym worktree | **tylko trening od zera**. Kod: `models/ariel_winner_trace_nf/` (zawiera zagnieżdżony `ariel_winner_nf/`) |
| **NSF, wariant źródłowy** (`ariel_winner_nf`) | — | — | — | jak `adc_winner_on_ariel` | — | snapshot źródłowy, bez własnego przebiegu | **tylko trening od zera**. Kod: `models/ariel_winner_nf/` |
| **FMPE na TauREx** (`taurex_fmpe`) | crossgen tau train | **`holdout` ≡ `validation`** (ten sam selektor) | **brak testu in-domain**; POSEIDON wykluczony jawnie | 218 binów | nie zmierzone | brak danych o przebiegu | **tylko trening od zera**. Kod: `models/taurex_fmpe/` |
| **FMPE Giordano** (`fmpe_giordano`) | — | — | — | 19 plików `.py`, pakiet nietracked w gicie | — | brak przebiegu w repo | **tylko trening od zera**. Kod: `models/fmpe_giordano/` |
| **SBI/FMPE na ADC** (`sbi_ariel_adc2023`) | — | — | — | — | — | brak przebiegu w repo | **tylko trening od zera**. Kod: `models/sbi_ariel_adc2023/` |
| **Wariant 5-kubitowy** (`five_qubit_exobiome`) | — | — | — | jak ExoBiome | — | brak przebiegu w repo | **tylko trening od zera**. Kod: `models/five_qubit_exobiome/` |
| **Port na sprzęt IQM Garnet** (`garnet_ariel_quantum_regression`) | wagi flagowego checkpointu | 2000 wierszy walidacji ADC | **zdrowy** | jak ExoBiome | jak flagowy | **ewaluacja** checkpointu na backend IQM | **nieuruchamialny**: kod w `models/garnet_ariel_quantum_regression/` (`checkpoint.py`, `runtime.py`) importuje nieistniejący moduł `models.ariel_quantum_regression` (3 z 3 importów, `a14`); notebook `garnet_port_tutorial.ipynb` w tym katalogu ma **wyczyszczone outputy**. Jedyny zapis wykonania: `archive/early_prototype_snapshot/ariel/models/notebooks/garnet_ariel_quantum_regression/benchmark_report_exobiome8_garnet.md` |
| **`12q_taurex_exobiome`** | — | — | — | — | — | **pakiet nie istnieje** — udokumentowany w `docs/12q_taurex_exobiome_architecture.md`, objęty testem `tests/test_12q_taurex_exobiome.py`, ale katalogu `models/12q_taurex_exobiome/` nie ma (`a14`) | **brak** |

---

## 2. Stan zbiorów danych

| zbiór | rozmiar | stan | dowód |
|---|---|---|---|
| **ADC2023** (`data/ariel-ml-dataset`) | 41 423 (split 33 138 / 4142 / 4143) | **zdrowy** — dane organizatorów challenge'u, pobrane, nie generowane u nas | poprawność zbioru jest własnością źródła; niezależnie od niej rekonstrukcja naszego pipeline'u przechodzi wiersz po wierszu wobec re-ewaluacji na CPU, 0 z 4143 wierszy powyżej tolerancji (`a27`, kolumna `mac_cpu`). `a01` nie był na tym zbiorze uruchamiany |
| **crossgen tau** | 41 423 (37 281 / 4142) | **niepełny fizycznie** — generator wywołany **bez CIA**, głównego kontinuum w atmosferze H₂/He | `taurex_backend.py:154-157`: `contributions` ma 2 elementy, brak `CIAContribution`; oficjalny FM organizatorów ma 3 (K11) |
| **crossgen POSEIDON** | 685 | **zepsuty** — 685/685 widm `transit_depth_noiseless` ma **jedną wartość na 218 binów**; pole czytane przez modele (`transit_depth_noisy`) ma `amp/σ = 0,998` i `SNR > 3` w **zerowej** liczbie wierszy | `a01`, kryterium bezprogowe |
| **pRT validation** (`petitradtrans-adc2023-validation`) | 20 000 | **nieużywany, nieoceniony** — zbiór nie jest wejściem żadnego modelu ani żadnego checku, więc jego stan nie jest w tym raporcie twierdzeniem | zero konsumentów poza własnymi skryptami generującymi (`data/prt_adc2023_validation/`); pełna inwentaryzacja w `docs/INWENTARYZACJA_DANYCH.md` §4 |

---

## 3. Ile z tego da się dziś odtworzyć

| co | ile pakietów | które |
|---|---:|---|
| **odtworzenie liczby z zapisanych predykcji** (bez wag, bez treningu) | 4, **ale nie w pełni** | `ariel_exobiome`, `adc_winner_on_ariel` — pełne; `taurex_exobiome`, `taurex_exobiome_without_quant` — **tylko strona POSEIDON**, odtworzona z predykcji (3,215615 bit-exact, 3,279559 z Δ ≈ 6e-7); strona tau/val (1,449002 / 1,423032) nie ma pliku metryk ani predykcji, tylko informacje README z jawnym „not yet re-derived" (`reports/taurex_model_comparison.md:22-23`) |
| **ponowna inferencja z wag** | 4 | `ariel_exobiome`, `taurex_exobiome`, `adc_winner_on_ariel`, `adc_baseline` |
| **wyłącznie trening od zera** | 6 | `ariel_winner_nf`, `ariel_winner_trace_nf`, `taurex_fmpe`, `fmpe_giordano`, `sbi_ariel_adc2023`, `five_qubit_exobiome` |
| **nieuruchamialne bez naprawy kodu** | 2 | `garnet_ariel_quantum_regression` (3 zepsute importy), `12q_taurex_exobiome` (pakiet nie istnieje) |

**Liczby cytowane bez pliku predykcji obok nich — 3 pozycje oznaczone wprost przez `a02`**: CNN 0,6500
(`cnn_metrics.json`, `NO backing artifact`), „winner on TauREx" 3,4531 (`results_summary.md`), H200
2,8946.

Nie jest to ta sama lista, co „liczby bez artefaktu" w Raporcie metodologicznym audytu:

| check | pytanie | wynik |
|---|---|---|
| `a02` | czy **obok cytowanej liczby** leży plik predykcji, z którego da się ją przeliczyć | 3: CNN 0,6500, winner-on-TauREx 3,4531, H200 2,8946 |
| `a13` | czy claim ma **gdziekolwiek w repo** wagi ∧ metryki ∧ predykcje | 4 `UNBACKED`: `random_forest_holdout`, `winner_on_taurex_poseidon`, `h200_poseidon`, `garnet_hardware` |


**Wagi zaimportowane, nieopisane:** `artifacts/imported_weights/` zawiera ~150 plików `.pt`
(m.in. `fmpe_outputs/`) **bez ani jednego pliku metryk, configu czy scalerów**. Z tego powodu nie da się ustalić,
któremu warunkowi eksperymentu odpowiada dany plik. Jedynym zachowanym zapisem tego przypisania są
notebooki w `archive/`.

---

## 4. Uwagi metodologiczne do inwentarza

1. **Tylko jeden model w całym repo ma pełny łańcuch dowodowy** (wagi + predykcje + metryki + config +
   historia + manifest splitu): flagowy `ariel_exobiome`. Wszystkie porównania międzymodelowe opierają
   się więc na pakietach o niejednakowej kompletności artefaktów.
2. **Sumy parametrów są zmierzone dla dwóch pozycji**, bo `a06` mierzy wyłącznie wariant flagowy
   i model odniesienia. Dla obu wariantów crossgen zmierzona jest natomiast **gałąź rezydualna** —
   `a11.residual_branch_params` liczy ją z `named_parameters` po obu stronach (`both_arms_measured =
   true`): ramię kwantowe **69 434** par. (z tego 24 kwantowe), klasyczna kontrola **85 733** par.,
   czyli **kontrola ma o 23,5 % więcej pojemności rezydualnej** niż ramię, które kontroluje, i nie ma
   bramki.
3. **Żaden pakiet nie przewiduje pełnego wektora celu benchmarku** — wszystkie mają 5 wyjść (gazy),
   benchmark ADC2023 wymaga 7 (`planet_radius`, `planet_temp` + 5 gazów). Wyjątkiem jest
   `ariel_winner_trace_nf`, który deklaruje 7, ale nigdy nie został wytrenowany.
4. **Dwa pakiety oceniają się na własnych danych walidacyjnych:** taurex_fmpe (SOURCE_HOLDOUT_SPLIT = SOURCE_VALIDATION_SPLIT = "val", models/taurex_fmpe/constants.py:19-21; self-udokumentowane jako holdout_mirrors_validation: true w prepare_dataset.py) i taurex_exobiome_without_quant (taurex_ignore_poseidon = true w zacommitowanym config.json runu → holdout_source_indices = val_source_indices.copy(), dataset.py:918-919, więc jego holdout_metrics.json to metryka walidacyjna, nie test na POSEIDON). Weryfikacja checkiem: audit/a10_split_integrity.py.

5. **Jeden „model" w tabeli nie jest modelem, a ewaluacją**: port na sprzęt IQM Garnet uruchamia wagi
   flagowego checkpointu na czterech backendach. Jest to jedyny w projekcie zapis wykonania na
   prawdziwym sprzęcie kwantowym — i jest wyłącznie w `archive/`, bo żywy notebook ma wyczyszczone
   outputy.
