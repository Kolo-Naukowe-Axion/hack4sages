# Protokół badawczy — retrieval biosygnatur egzoplanetarnych (ExoBiome)

## 0. Cel i zakres

Zadanie badawcze: retrieval pięciu abundancji gazowych (H₂O, CO₂, CO, CH₄, NH₃) z widm transmisyjnych
egzoplanet, przy użyciu hybrydowej architektury kwantowo-klasycznej, oceniany na dwóch osiach:

1. **Oś ADC2023** — dane organizatorów wyzwania Ariel Data Challenge 2023, split organizatorów.
2. **Oś cross-generator** — model wytrenowany na jednym generatorze fizycznym (TauREx) i oceniany na
   innym (POSEIDON), jako test transferu.

Protokół obejmuje: 
- (§1) środowiska, 
- (§2) charakterystykę danych oraz generatorów,
- (§3) podział zbiorów, 
- (§4) konwencję przygotowania wejścia,
- (§5) schemat treningu, 
- (§6) dobór metryki wraz z konsekwencjami projektowymi, 
- (§7) warunki odtwarzalności wyniku 
- (§8) oraz procedurę weryfikacji, przez którą liczba musi przejść, zanim zostanie nazwana faktem, 
- (§9) co protokół pomija
- literatura
**Nie** obejmuje wyboru architektury sieci (opisanej w kodzie modeli) ani interpretacji naukowej wyników
(sekcje K audytu).

**Jak czytać ten dokument.** Protokół jest sformułowany jako zbiór warunków, pod
którymi to badanie jest ważne. Nie odnosi się do tego, co zostało zrobione. Każda reguła ma jednak uzasadnienie w pomiarze z tego repozytorium; przy regułach podane są odsyłacze do ustaleń
audytu (`K1`, `K3`, …), które pokazują, co się dzieje, gdy reguła nie jest zachowana.

Część wykonywalna protokołu to kod w `audit/`: odtwarza **ustalenia tego audytu na tych artefaktach**.
Nie jest to harness, którym da się ocenić dowolny nowy wynik — to narzędzie regresyjne nad zamrożonym
zestawem artefaktów i jego granice opisuje raport końcowy, nie ten protokół.

## 1. Środowiska

Nie ma jednego wspólnego środowiska, bo zależności kolidują. Do odtworzenia całości potrzeba **sześciu**:
dwóch dla audytu i modeli, jednego dla diagnostyki POSEIDON, i trzech dla generatorów danych. Środowisko
diagnostyczne i generator POSEIDON mają tę samą wersję biblioteki (1.3.2), ale inne zestawy zależności,
więc liczone są osobno.

**Audyt i modele:**

| środowisko | python | zastosowanie | plik pinów |
|---|---|---|---|
| `.venv-qml` | 3.12 | modele hybrydowe (torch + PennyLane 0.44 + zuko), cały `audit/` | `requirements-qml.txt` |
| `.venv-cnn` | 3.11 | baseline CNN organizatorów (TensorFlow 2.21 + Keras 3 + taurex 3.3.2) | `requirements-cnn.txt` |

**Diagnostyka POSEIDON (`d01`):**

| środowisko | wersja POSEIDON | rola | pin |
|---|---|---|---|
| venv budowany pod `d01` z pinu `POSEIDON @ git+...@v1.3.2`, python 3.11.9 | **1.3.2** | **wersja generująca zbiór** (`manifest.json: software_versions.poseidon`) — diagnoza Etapu 1 (K1(c)) została wykonana **na tej wersji**; rekord: `reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json` (`env.python = 3.11.9`, `payload.stage0.poseidon_version = 1.3.2`) | komenda pełna w `audit/README.md` i bloku weryfikacyjnym K1(c) |

**Generacja danych:**

| plik | generator | wersja przypięta |
|---|---|---|
| `requirements-generator-poseidon.txt` | strona POSEIDON crossgen | `POSEIDON @ ...@v1.3.2` + `mpi4py`, `setuptools<81` (deklarowane zależności systemowe: `open-mpi`) |
| `requirements-generator-taurex.txt` | strona tau crossgen | `taurex==3.2.4` |
| `requirements-generator-prt.txt` | generator pRT validation | `petitRADTRANS==2.6.7`, `taurex==3.2.4`, python 3.9.x, reszta pinów jak w `manifest.json` |

Każdy plik odtwarza dokładnie tę wersję (mogą istnieć nowe wersje), która wygenerowała odpowiadające dane w repo
(`manifest.json: software_versions`), więc regeneracja z nich jest tym samym eksperymentem, nie
przybliżeniem.


## 2. Dane: generatory i pochodzenie

### 2.1 O generatorach

**TauREx 3** (Waldmann i in. 2014, TauREx I, arXiv:1409.2312; Al-Refaie i in. 2021, TauREx 3,
arXiv:1912.07759) — otwarty kod transferu promienistego dla widm transmisyjnych i emisyjnych,
budowany modularnie z „wkładów" (`Contribution`): absorpcja molekularna, rozpraszanie Rayleigha,
CIA, chmury. Każdy z nich dołączany jest osobno do listy `contributions` przekazanej do modelu. Ta modularność
jest źródłem K11: pominięcie wkładu nie rzuca błędu, tylko cicho zmienia fizykę widma. TauREx jest
silnikiem generującym ADC2023.

**POSEIDON** (MacDonald, JOSS 2023, doi:10.21105/joss.04873; arXiv:2410.18181) — kod modelowania
i retrievalu, natywnie wielowymiarowy (1D/2D/3D), rozwijany pod obserwacje JWST/HST.

**petitRADTRANS** (Mollière i in. 2019, arXiv:1904.11504, A&A 627, A67) — kod transferu promienistego
z metodą correlated-k (niska rozdzielczość, R=400 natywnie) i line-by-line (wysoka rozdzielczość),
jedyny z trzech używany tu z chmurami (`MgSiO3(c)_cd`, `Fe(c)_cd`).

#### Porównanie możliwości trzech generatorów
| własność | TauREx 3 | POSEIDON | petitRADTRANS |
|---|---|---|---|
| **profile T–P** | izoterma, Guillot (2010) dwustrumieniowy, Rodgers (2000) layer-by-layer, N-punktowy, z pliku | **9 opcji**: `isotherm / gradient / two-gradients / Madhu / Pelletier / Guillot / Line / slope / file_read` | wielopunktowy (w naszej konfiguracji: `t1/t2/t3/t_connect/t_int`) |
| **chemia** | równowagowa (ACE Fortran, C/O + metaliczność) albo swobodna per molekuła (stała / dwuwarstwowa / z pliku) | `isochem / gradient / two-gradients / lever / file_read / chem_eq` | równowagowa z quenchingiem (`c_o`, `fe_h`, `log_p_quench`, `log_kzz`) |
| **chmury** | **tak**: Mie (kilka implementacji) + szare chmury — jako **opcjonalny wkład** | **tak**: `cloud-free / MacMad17 / Iceberg / Mie`, typy `deck / haze / deck_haze` | **tak**: prawa potęgowe, stałe optyczne realnych kondensatów, ziarna amorficzne/krystaliczne, sferyczne/nieregularne |
| **CIA** | **opt-in** — trzeba jawnie dodać `CIAContribution` | **automatyczne** — `CIA_pairs` budowane z listy gazów | dostępne |
| **wymiarowość** | 1D (+1.5D w rozszerzeniu PhaseCurve) | **1D / 2D / 3D natywnie** | 1D |
| **rozdzielczość** | siatka natywna = najwyższa rozdzielczość wczytanego opacity; rebinowanie klasą `FluxBinner` | `wl_grid_constant_R(wl_min, wl_max, R)` — stałe R, jednorodne w `log(wl)` | **dwa tryby**: correlated-k **R=1000**, line-by-line **R=10⁶** |
| **szum obserwacyjny** | **liczy σ, ale nie losuje** — `SNRInstrument` zwraca σ obok czystego widma; realizacja należy do użytkownika (szczegóły niżej) | **liczy σ i losuje** realizację (`np.random.normal`, pod flagą `Gauss_scatter`) | brak wbudowanego modelu instrumentu |
| **wyjście** | transmisja: głębokość tranzytu; emisja: intensywność | konfigurowalne: `(Rp/Rs)^2 / Rp/Rs / Fp/Fs / Fp` (`core.py:2167`) | transmisja **i** emisja |

Informacje zweryfikowane poprzez:
- **TauREx 3** — publikacja Al-Refaie i in. 2021 ([arXiv:1912.07759](https://arxiv.org/abs/1912.07759),
  ApJ 917, 37, [doi:10.3847/1538-4357/ac0252](https://iopscience.iop.org/article/10.3847/1538-4357/ac0252))
  oraz przegląd kodu: [ucl-exoplanets/taurex3](https://github.com/ucl-exoplanets/taurex3), katalog
  `src/taurex/instruments/` i `src/taurex/contributions/`
- **POSEIDON** — przegląd kodu przypięty do wersji, którą generowaliśmy dane:
  [tag v1.3.2](https://github.com/MartianColonist/POSEIDON/tree/v1.3.2); opcje `PT_profile`,
  `X_profile`, `cloud_model` w docstringu `define_model()`
  ([core.py#L421-L429 w v1.4.0](https://github.com/MartianColonist/POSEIDON/blob/594f6f5632/POSEIDON/core.py#L421-L429));
  publikacja: MacDonald, JOSS 2023 ([doi:10.21105/joss.04873](https://joss.theoj.org/papers/10.21105/joss.04873),
  [arXiv:2410.18181](https://arxiv.org/abs/2410.18181))
- **petitRADTRANS** — publikacja Mollière i in. 2019
  ([arXiv:1904.11504](https://arxiv.org/abs/1904.11504), A&A 627, A67); rozdzielczości R=1000
  (correlated-k) i R=10⁶ (line-by-line) oraz traktowanie chmur podane wprost w abstrakcie.
  Uwaga: używamy **2.6.7**, a aktualna linia to **3.x** (Blain i in. 2024, breaking changes)

**Szum: kto liczy σ, a kto losuje realizację**

Generator ma szum może znaczyć
dwie różne rzeczy: że **wylicza wielkość** niepewności (σ), albo że **dolosowuje konkretną realizację**
i dodaje ją do widma.

| | wylicza σ | losuje realizację | uwaga |
|---|---|---|---|
| **TauREx 3** | **tak** — `SNRInstrument` liczy `σ = (max−min)/SNR`, `InstrumentFile` czyta σ z pliku | **nie** | `model_noise()` zwraca *czyste* widmo **i osobno** tablicę σ; w całym pakiecie `taurex/instruments/` nie ma ani jednego wywołania `np.random` — realizację losuje użytkownik |
| **POSEIDON** | tak (`syn_err`, precyzja w ppm) | **tak**, opcjonalnie | `err = np.random.normal(0.0, syn_err)`, dodawane do widma, pod flagą `Gauss_scatter` |
| **petitRADTRANS** | brak wbudowanego modelu instrumentu | nie | publikacja nie wspomina o szumie; w naszej konfiguracji σ i realizację dokłada nasz skrypt |

Weryfikacja: [`taurex3 src/taurex/instruments/snr.py#L48-L81`](https://github.com/ucl-exoplanets/taurex3/blob/7d9168609d4366430602599c8a4b9a78f4c7c35f/src/taurex/instruments/snr.py#L48-L81)
— `model_noise()` liczy σ w linii 79 i zwraca je **obok** niezmienionego widma; w żadnym z czterech
plików `src/taurex/instruments/` nie ma `np.random`. Dla porównania
[`POSEIDON/instrument.py#L576-L577`](https://github.com/MartianColonist/POSEIDON/blob/594f6f5632/POSEIDON/instrument.py#L576-L577)
— tam realizacja jest losowana i dodawana.

**Konsekwencja dla protokołu:** jeśli generator zwraca σ, ale nie losuje, to **decyzja o realizacji
szumu należy do nas** i musi być zapisana jako część konwencji zbioru (ISTOTNE §7.1: seed losowania szumu).

### 2.2 Zastosowane w projekcie i błędy

Projekt używa **trzech** źródeł widm (czwarte, wygenerowane źródło petitRADTRANS nie jest używane przez żaden model), o różnej naturze i różnym stopniu poprawności generacji (pełne
ustalenia i dowody: `docs/INWENTARYZACJA_MODELI.md` §2, oraz K1/K1(c)/K11 audytu):

| zbiór | generator | wersja | wiersze | rola w protokole |
|---|---|---|---|---|
| **ADC2023** (`data/ariel-ml-dataset`) | TauREx (organizatorzy) | — (dane zewnętrzne, nie generujemy sami) | 41 423 (33 138 / 4 142 / 4 143) | oś główna: trening i ewaluacja modelu flagowego |
| **crossgen tau** | TauREx | **3.2.4** (`manifest.json: software_versions.tau`) | 41 423 (37 281 / 4 142) | trening modeli cross-generator; oś transferu (strona źródłowa) |
| **crossgen POSEIDON** | POSEIDON | **1.3.2** (`manifest.json: software_versions.poseidon`) | 685 | oś transferu (strona docelowa) — **wyłącznie test** |

**Generator ADC2023, ustalone z kodu, nie z domysłu.** Oficjalny kod forward modelu organizatorów,
zawendorowany w `models/adc_baseline/FM_utils_final.py`, importuje bezpośrednio z pakietu `taurex`:
`AbsorptionContribution`, `RayleighContribution`, **`CIAContribution`** (`:174`), `BlackbodyStar`
(`:175`), `FluxBinner` (`:158`).

Oba ramiona crossgen pochodzą z **jednego** przebiegu generatora, `data/crossgen_biosignatures/`
(`MASTER_SEED = 20260310`, `manifest.json: seed`), wspólne parametry: promień planety,
`log g`, temperatura, promień gwiazdy i pięć log-VMR losowane z tych samych rozkładów po obu stronach
(`constants.py:28-33`), następnie przepuszczane przez **dwa różne** backendy fizyczne
(`tau_backend.py`, `poseidon_backend.py`). Siatka długości fali jest wspólna: 218 binów, rozdzielczość
stała R=100, zakres 0,6–5,25 µm (`manifest.json: wavelength_grid`) — **inna** niż siatka ADC2023/pRT
(52 biny, 0,55–7,28 µm, `petitradtrans-adc2023-validation/manifest.json: canonical_output_wlgrid`).

Trzy różnice między tau i POSEIDON wewnątrz crossgen, z regułami które z nich wynikają:

| różnica | stan | reguła |
|---|---|---|
| **TauREx wywołany bez CIA** — `contributions` ma 2 elementy zamiast 3 (`taurex_backend.py:154-157`) | widma tau fizycznie niepełne (K11) | żadne zdanie o transferze nie jest ważne, dopóki strona źródłowa nie zostanie zregenerowana z CIA — warunek (1) w §2.3 |
| **POSEIDON policzony bez bazy opacji** — `opacity_files = []`, `input_data_root = null` (`d01` stage 1) | widma POSEIDON stałe po λ, 685/685 wierszy (K1, K1(c)) | strona docelowa wymaga wczytanej bazy przekrojów — warunek (2) w §2.3 |



### 2.3 Jakie generatory można ze sobą łączyć - **żadnych dwóch bezpośrednio, bez harmonizacji**.

**Do retrievalu bayesowskiego generatory są w praktyce wymienne.** Dla WASP-39 b pięć różnych modeli
opacity zastosowanych do tego samego widma JWST dało abundancje rozrzucone o **0,13 dex** (H₂O),
**0,18 dex** (CO₂) i **0,27 dex** (SO₂) — wszystkie w granicach słupków błędu, z wnioskiem autorów, że
silne absorbery da się wiązać w granicy ~0,30 dex ([arXiv:2303.03383](https://arxiv.org/html/2303.03383v2)).
Dopasowanie z wolnymi parametrami absorbuje różnicę implementacji.

**Do treningu modelu ML nie są.** Model uczy się konkretnego odwzorowania widmo → parametry, razem
z artefaktami swojego generatora. Skala rozbieżności: w intercomparison MALBEC różnice między kodami
sięgały **500 ppm przy 2 µm** dla kodów bez algorytmu sub-layeringu
([arXiv:2402.04329](https://arxiv.org/abs/2402.04329), test T0A) — przy naszym σ = 20–100 ppm to
wielokrotność szumu obserwacyjnego, czyli widmo z innego kodu leży dla wytrenowanego modelu poza
rozkładem treningowym. Zastrzeżenie: 500 ppm to górna granica jednego efektu w jednym teście,
niwelowana przez sub-layering — nie wolno tego cytować jako „generatory różnią się o 500 ppm".

To właśnie czyni oś cross-generator **sensownym testem**: mierzy, czy model nauczył się fizyki, czy
artefaktów kodu. Ale test jest ważny tylko wtedy, gdy obie strony są fizycznie kompletne.

Jeśli któreś z generatorów mają zostać połączone w jeden zbiór `crossgen`, trzeba dopilnować:

1. **Kazdy z datasetów spełnia te same fizyczne zalezności** - np. CIA, poziom ciśnienia, zastosowane augumentacje.
2. **Test in-domain po obu stronach** — osobny, nieużywany w treningu split dla kazdego generatora. Bez tego nie da się rozdzielić „model nie przenosi się" od „model nie działa wcale".
3. **Identyczna konwencja wejścia** po obu stronach: ta sama normalizacja, ta sama decyzja
   o dolosowywaniu szumu, ta sama lista cech pomocniczych.
4. **Zrównany budżet treningowy** — ta sama liczba wierszy, epok i harmonogram LR dla każdego
   porównywanego wariantu.
5. **Baseline na obu osiach** — podłoga stałej treningowej policzona osobno dla każdego generatora.


## 3. Podział na zbiory

### 3.1 W naszym projekcie

| oś | train | val | test/holdout | metoda podziału |
|---|---:|---:|---:|---|
| ADC2023 | 33 138 | 4 142 | 4 143 | podział organizatorów, stratyfikacja `presence_signature` (`split_manifest.json: primary_stratify_mode`), `split_seed = 42` |
| crossgen tau | 37 281 | 4 142 | — (brak testu in-domain) | podział generatora, zapisany w `manifest.json: generator_summary.tau.split_counts` |
| crossgen POSEIDON | — | — | 685 | całość jako test|


## 4. Konwencja przygotowania wejścia

To jest miejsce, w którym projekt ma **udokumentowaną niespójność** (K3 audytu) — protokół poniżej
opisuje obie konwencje i który wariant jest właściwy dla którego porównania, żeby nie powtórzyć błędu.

**Cechy wejściowe, ADC2023 (model flagowy `ariel_exobiome`):**
- 52 biny `instrument_spectrum` (widmo jak w pliku, **bez dolosowanego szumu**)
- 52 `instrument_noise`
- `instrument_width`, `wavelength_um`
- 8 kolumn aux: `star_distance`, `star_mass_kg`, `star_radius_m`, `star_temperature`,
  `planet_mass_kg`, `planet_orbital_period`, `planet_distance`, `planet_surface_gravity`
  (`models/ariel_exobiome/constants.py:8-17`)
- normalizacja: dzielenie widma przez jego własną średnią (per próbka, nie globalnie)

**Cechy wejściowe, model porównawczy NSF (`adc_winner_on_ariel`):**
- 52 biny widma + **szum dolosowywany co epokę** (`sample_noise=True`, `preprocessing.py:148-149`),
  standaryzacja **per widmo** (mean 0, std 1 po binach, `preprocessing.py:153-155`), a odjęta średnia
  i odchylenie wchodzą do modelu jako dwie osobne cechy
- 4 estymatory promienia jako dodatkowe cechy

**Reguła obowiązująca dla każdego przyszłego porównania między tymi dwoma konwencjami:** Każde przyszłe porównanie modeli **musi** albo używać identycznej
konwencji wejścia po obu stronach, albo jawnie raportować obie wersje modelu (czysta / zaszumiona) zamiast
jednej.

**Cechy wejściowe, oś crossgen (oba warianty `taurex_exobiome*`):** 218 binów `transit_depth_noisy`
(TauREx) — ta sama receptura co ADC, ale na innej siatce spektralnej.

## 5. Trening

Schemat dwuetapowy (dotyczy modelu flagowego i wariantów crossgen; szczegóły liczbowe per model w
`docs/INWENTARYZACJA_MODELI.md`):

1. **Etap 1 — backbone klasyczny.** Trening wyłącznie ścieżki klasycznej, checkpoint
   `stage1_classical/best_model.pt`.
2. **Etap 2 — adapter kwantowy.** Ładowany checkpoint z etapu 1 (`init_checkpoint_path`), backbone
   **zamrożony przez `quantum_backbone_freeze_epochs` epok**, w tym czasie uczy się wyłącznie adapter
   kwantowy (`quantum_gate`, `quantum_head`). `quantum_scale` narasta liniowo z `quantum_warmup_epochs`
   do 1.0 w ciągu `quantum_ramp_epochs` epok (`resolve_quantum_scale`, `training.py:335-341`).
   Po `quantum_backbone_freeze_epochs` backbone się odmraża.

**Zasada dla przyszłych treningów, wynikająca z K4/K5:** jeśli checkpoint jest wybierany przed końcem
rampy `quantum_scale` (czyli przed epoką `quantum_warmup_epochs + quantum_ramp_epochs`), **raportowana
metryka musi być liczona przy tej samej skali, przy której checkpoint był wybrany**, nie przy domyślnej
`quantum_scale=1.0` sygnatury funkcji. Ewaluacja przy skali, której model nigdy nie widział w treningu nie jest pomiarem. Każdy nowy trening powinien albo 
- (a) domykać rampę do końca przed selekcją checkpointu 
- albo (b) zapisywać `quantum_scale` selekcji w `config.json` obok wag i używać go konsekwentnie przy każdej późniejszej ewaluacji tego checkpointu.

### 5.1 Trening wieloseedowy

Stan obecny: **jeden seed w całym projekcie** (`a12`: wartość 42, znaleziona w 32 polach seedowych
w 112 artefaktach). Każde zdanie o istotności
w tym repo pochodzi z **bootstrapu po wierszach testowych jednego checkpointu**. To szacuje precyzję pomiaru dla **jednego
wytrenowanego modelu**, a to nam nie mówi nic o tym, czy model wytrenowany ponownie pokazałby ten sam efekt.

**Ile seedów.** Kryterium `a12` wymaga **≥ 5** seedów na ramię. To minimum, przy którym da się
policzyć rozrzut międzyseedowy z sensowną liczbą stopni swobody, nie liczba dobrana po wyniku
(`SEEDS_REQUIRED = 5`, `a12:48`, zadeklarowana przed pomiarem).

**Cały wkład ścieżki kwantowej** to
**0,0069** mRMSE przy skali selekcji i **0,0037** przy skali 1,0 (`a04`, ADC holdout, n = 4143). MDE dla
n = 4143 nie jest w repo policzone, nikt nie zmierzył.

Wniosek: **efekt kwantowy jest
o rząd wielkości mniejszy od progu wykrywalności na tych rozmiarach zbiorów.** Wieloseedowość jest
konieczna, żeby twierdzenie architektoniczne było w ogóle dobrze postawione.

**Minimalna procedura, jeśli twierdzenie architektoniczne ma zostać postawione:**

1. **≥ 5 seedów na ramię**, seed zapisany w `config.json` każdego przebiegu (dziś `a12` znajduje go po
   nazwie pola, więc konwencja nazewnicza musi zostać).
2. **Delta liczona per seed**, nie na sklejonych predykcjach — para (seed, ramię A) ↔ (seed, ramię B)
   z tym samym seedem po obu stronach, żeby różnica nie mieszała wariancji inicjalizacji z efektem.
3. **Przedział ufności po seedach**, nie po wierszach.
4. **Korekta na wielokrotność zadeklarowana z góry.**
5. **Zrównany budżet między ramionami**

Weryfikacja: `./.venv-qml/bin/python audit/a12_significance_power.py`. Check przechodzi z `FAIL` na
`PASS` dopiero wtedy, gdy oba kryteria są spełnione: liczba seedów **i** efekt większy od
rozrzutu międzyseedowego. Dziś drugi człon jest jawnie **nieoceniany** (`effect_exceeds_seed_spread:
null`), bo przy jednym seedzie rozrzutu nie da się policzyć; nie jest zaliczany ani na korzyść, ani na
niekorzyść.

## 6. Ewaluacja i metryki

### 6.1 Jakie metryki są używane w tym zadaniu

Przegląd literatury z pełnymi cytowaniami na końcu pliku. Wnioski istotne dla protokołu:

| konwencja | metryka | uwaga |
|---|---|---|
| **ADC2023** | `0,8 × score_posterior + 0,2 × score_spectral`; `score_posterior` = dwupróbkowy **test Kołmogorowa–Smirnowa** na 7 rozkładach marginalnych wobec próbek referencyjnych MultiNest; `score_spectral` = odwrócona strata Hubera na medianie widma i IQR | spec potwierdzony trzema niezależnymi pracami uczestników; **brak oficjalnego papieru** |
| **prace SBI/FMPE** (np. arXiv:2312.08295) | log-likelihood / importance sampling na posteriorze, kalibracja pokrycia | wymagają wyjścia rozkładowego |
| **RMSE / mRMSE punktowe** | używane jako metryka pomocnicza | **nie występuje** jako metryka główna w żadnej edycji challenge'u |

**Reguła wynikająca z tego przeglądu:** w tym zadaniu standardem dziedziny jest metryka **rozkładowa**.
Estymator punktowy oceniany na mRMSE jest oceniany na skali, której benchmark nie używa i to nie jest
kwestia surowości, a **różnicy kategorii obiektu wyjściowego**.

**Metryka używana w tym projekcie: mRMSE** (średnia RMSE po pięciu gazach, w jednostkach log₁₀ VMR).
Nie jest metryką ADC2023, ale z dwóch powodów.

**(a) KS na modelu punktowym: policzyć się da, tylko wynik mierzy co innego.** Estymator punktowy
wchodzi do testu KS jako **delta Diraca** — to jest legalny rozkład, więc statystyka istnieje i `a24`
ją policzył: `ks.mean.exobiome = 0,7508` na 663 planetach (per gaz 0,710–0,784). Rzecz w tym, że
delta Diraca ma wobec dowolnego ciągłego rozkładu referencyjnego **KS ≥ 0,5 z konstrukcji**, przy
podłodze skończonej próbki 0,044. Liczba jest więc zdominowana przez **niezgodność typu obiektu
wyjściowego**, nie przez jakość modelu: dokładając niepewność do tych samych predykcji poprawiłoby
się KS, nie zmieniając ani jednej wartości centralnej. Dokładnie to znaczy „tracimy plusy
posteriora" — kalibrację, szerokość i wielomodalność.

Dwa zastrzeżenia, żeby nie przesadzić w drugą stronę:

- **„≥ 0,5" nie znaczy „najgorszy możliwy".** Ramię PRIOR (rozkład ignorujący widmo) dostaje
  `0,6101`, czyli **gorzej** niż jakikolwiek punkt trafiający w środek posterioru. Dlatego kryterium
  `a24` jest falsyfikowalne również wewnątrz klasy modeli punktowych. ExoBiome dostaje `0,7508`,
  czyli skill wobec prioru **−0,2305** — to jest realny, informatywny pomiar, nie artefakt.
- **Nie każda metryka rozkładowa tak działa.** Na Wasserstein-1 i na light-tracku ten sam estymator
  punktowy **bije** ramię PRIOR (skill **+0,4459** i **+0,5384**). Zdanie „modelu punktowego nie da
  się ocenić rozkładowo" byłoby więc za mocne — problem dotyczy **KS**, czyli członu o wadze 0,8.

**(b) Pięć wyjść wobec siedmiu marginalnych — i to jest przeszkoda twarda.** `score_posterior` jest
zdefiniowany jako KS na **7** rozkładach marginalnych. Przy 5 wyjściach dwóch członów sumy nie ma
z czego policzyć, więc oficjalnego score'u nie da się wyprodukować **w ogóle**, niezależnie od tego,
jaką głowę niepewności się dołoży. `a24` odnotowuje to wprost:
`not_scored = [planet_radius, planet_temp, ADC2023 spectral component]`.

**Każde przyszłe porównanie z wynikami zgłoszonymi na ADC2023 (w tym z pracą Aubin i in. 2023) musi
jawnie zaznaczyć, że to porównanie międzymetrykowe, nie ranking na tej samej skali.**

### 6.2 Konsekwencja projektowa: równa ewaluacja wymaga głowy niepewności

ExoBiome w obecnej formie jest **deterministycznym regresorem punktowym** na pięć liczb, jedna
wartość każda. Skutki:

1. **Nie potrafi wyrazić struktury, którą dane faktycznie mają.** Posterior `(T, log X)` jest w tym
   zadaniu opisywany w literaturze jako wielomodalny i silnie zdegenerowany. Punkt musi uśrednić po
   modach i może wypaść w obszarze, który dane aktywnie odrzucają.
2. **Wynik jest niefalsyfikowalny w sensie fizycznym.** Bez `T` i `R_ref` nie ma czego podstawić do
   forward modelu, więc nie ma widma resztkowego, χ², ani evidence. Nie da się odróżnić „widmo
   ogranicza H₂O do −4,9 ± 0,3" od „widmo było płaskie i model zwrócił średnią prioru".
**Zalecenie:** równa ewaluacja wobec zgłoszeń ADC2023 wymaga **dwóch zmian naraz**, i to jest
istotne — żadna z nich osobno nie wystarcza (§6.1: głowa rozkładowa zdejmuje podłogę Diraca na KS,
ale nie tworzy brakujących dwóch marginalnych; siedem wyjść bez rozkładu dalej wchodzi do KS jako
delta).

1. **Wyjście rozkładowe.** Dwie drogi, obie udokumentowane w tym repo:
   - głowa niepewności doklejona do istniejącego backbone'u. Baseline organizatorów robi to
     jednym argumentem: `Dropout(p)(x, training=True)` aktywny w inferencji
     (`models/adc_baseline/MCDropout.py:17,33,35`), na CNN o 2,3 MB wag. To dolna granica kosztu.
   - przeprojektowanie na estymator rozkładowy (normalizing flow / FMPE), czyli architektura
     z rodziny, którą wygrało ADC2023.
2. **Siedem wyjść zamiast pięciu** (`planet_radius`, `planet_temp` + 5 gazów) — bez tego
   `score_posterior` pozostaje nieliczalny z definicji, patrz K9 i `a24.not_scored`.

Do czasu **obu** tych zmian **wszystkie liczby ExoBiome wolno raportować wyłącznie jako mRMSE
z jawnym zastrzeżeniem, że to nie jest metryka challenge'u**, i nie wolno ich zestawiać z wynikami
zgłoszeń ADC2023 jako rankingu.

### 6.3 Wymóg baseline'u

Zanim jakikolwiek wynik zostanie ogłoszony jako „model coś umie", musi być zestawiony z **podłogą** —
mRMSE stałej równej średniej treningowej, na tych samych wierszach i tym samym generatorze
(`audit/a02_trivial_baseline.py`). Skill liczymy jako `1 − raportowane / podłoga`.

Trzy zasady stosowania, każda wynikająca z błędu popełnionego w tym repo:

1. **Podłoga liczona z etykiet treningowych**, nie ewaluacyjnych — stała ze zbioru ocenianego to
   oracle, nie baseline.
2. **Kryterium stosowane symetrycznie do wszystkich porównywanych wariantów.**
3. **Metryki zestawione z minimalnym efektem wykrywalnym** (`audit/a12_significance_power.py`).
   Dla przykładu na 685 wierszach MDE ≈ 0,044 **dex mRMSE** — to jednostka `a12`, nie skill; żeby
   porównać ją ze skillem, trzeba ją podzielić przez podłogę tego zbioru (2,8940 → ≈ 0,015 skillu).
   Różnicy mniejszej nie wolno rankować.

**Konwencja `quantum_scale` przy raportowaniu.** Każda liczba prezentowana jako wynik modelu z gałęzią
kwantową musi podawać, przy jakiej wartości `quantum_scale` została policzona, i ta wartość musi być
odtwarzalna z zapisanego artefaktu.

## 7. Warunki odtwarzalności


### 7.1 Co musi być zapisane obok każdej liczby

| element | dlaczego |
|---|---|
| **rewizja repozytorium** (`git_revision`) | kod modelu i preprocessing zmieniają się między przebiegami; bez tego nie wiadomo, która wersja liczyła |
| **sygnatura czasowa UTC** (`timestamp_utc`) | rozstrzyga kolejność przebiegów; nazwa katalogu ani data modyfikacji pliku **nie są** wiarygodne |
| **suma kontrolna wejść** (`sha256`) | dane mogą się podmienić bez zmiany nazwy pliku |
| **wersje bibliotek** | wynik zależy od wersji generatora, `torch`, `PennyLane` — §1 |
| **wszystkie seedy** | seed podziału, seed treningu, seed losowania szumu, seed próbkowania posterioru; każdy osobno |
| **punkt pracy modelu** | dla modeli z parametrem skalującym (np. `quantum_scale`) — wartość, przy której liczba została policzona, **nie** wartość domyślna w sygnaturze funkcji |

### 7.2 Trzy warunki, każdy konieczny

1. **Prowenienacja jest kompletna.** Wszystkie pola z §7.1 zapisane w rekordzie towarzyszącym liczbie,
   a nie odtwarzane z pamięci albo z nazwy katalogu.
2. **Liczbę da się przeliczyć od surowych danych.** Niezależna rekonstrukcja ścieżki wejściowej, daje tę samą wartość w granicy **jawnie podanej**
   tolerancji. „W przybliżeniu" nie jest tolerancją; tolerancja to liczba z uzasadnieniem, skąd się wzięła.
3. **Każde losowanie jest zaseedowane i zapisane.** Jeśli w ścieżce ewaluacji występuje próbkowanie
   (np. losowanie z posterioru modelu rozkładowego), seed musi być zapisany w rekordzie.

### 7.3 Artefakt modelu musi być kompletny
Żeby liczba z modelu była odtwarzalna, obok wag muszą leżeć:
- **konfiguracja przebiegu** — hiperparametry, harmonogramy, punkt pracy;
- **skalery / statystyki normalizacji** — wyliczone na splicie treningowym;
- **manifest splitu** — które wiersze były treningiem, walidacją, testem;
- **predykcje** na zbiorze, z którego pochodzi raportowana metryka;
- **historia treningu** — przebieg metryki po epokach, jeśli selekcja checkpointu była na niej oparta.

Rozróżnienie, które warto utrzymać przy
raportowaniu: *odtworzenie liczby z zapisanych predykcji* (nie wymaga wag), *ponowna inferencja z wag*
(nie wymaga treningu) i *ponowny trening od zera* to trzy różne poziomy, nie jeden.

## 8. Protokół weryfikacji wyniku

Ta sekcja opisuje procedurę, przez którą musi przejść liczba, zanim zostanie w raporcie nazwana faktem.

### 8.1 Dwa poziomy weryfikacji

**Poziom 1 — falsyfikowalność ustalenia (obowiązkowy dla każdego twierdzenia w raporcie).** 
Każda liczba, ustalenie, metryka i porównanie muszą mieć zapisane wymienione źródła (a) gdzie zostało znalezione, z cytatem `plik:linia`; (b) który check je mierzy i jaką komendą się go uruchamia.

**Poziom 2 — recenzja niezależna (obowiązkowa przed oddaniem).** Przegląd wykonany przez stronę, która
nie brała udziału w pisaniu ustaleń, ze szczególnym skupieniem na szukaniu błędów.

### 8.2 Klasy błędów, które ta procedura ma wychwytywać
Czyli przykłady tego, jakie błędy można było znaleźć w repo.

| problem | opis | jak wychwycić |
|---|---|---|
| rozbieżność proza ↔ payload | liczba w tekście nie zgadza się z polem JSON, na które się powołuje | porównanie|
| zły cytat kodu | `plik:linia` wskazuje inną treść niż ta, którą ma uzasadniać | otwarcie każdej cytowanej linii |
| fałszywa niezależność | wynik nazwany „niezależnym potwierdzeniem", gdy dzieli założenie ze potwierdzanym | sprawdzenie, czy obie drogi nie zakładają tego samego |
| próg bez uzasadnienia | tolerancja albo granica PASS przyjęta bez wyprowadzenia | sprawdzenie odporności wniosku na zmianę progu |
| niedopasowana próba | porównanie na różnych wierszach, splitach albo budżetach treningowych | porównanie `n_train`/`n_eval`/splitów po obu stronach |
| efekt pod progiem wykrywalności | różnica rankowana, choć mniejsza od MDE | zestawienie każdej różnicy z MDE dla tej wielkości próby |
| liczba bez artefaktu | wartość cytowana bez wag/predykcji/kodu, który ją produkuje | indeks prowenienacji (`a13`) |

## 9. Co ten protokół celowo pomija

Trzy rzeczy są poza zakresem świadomie, nie przez przeoczenie:

1. **Wybór i uzasadnienie architektury sieci.** Protokół nie
   rozstrzyga, czy architektura hybrydowa kwantowo-klasyczna jest do tego zadania właściwa. Jedyny
   wyjątek to §6.1–6.2 — i on nie dotyczy architektury jako takiej, a **klasy obiektu wyjściowego**
   (punkt vs rozkład), bo od niej zależy, czy metryka benchmarku jest w ogóle policzalna.
2. **Interpretacja naukowa wyników.** To treść ustaleń K1–K11 audytu, nie protokołu.
3. **Ocena jakości pracy**

---

## Literatura

**Generatory (forward modele):**

- Waldmann i in. 2014 — *TauREx I* ([arXiv:1409.2312](https://arxiv.org/abs/1409.2312))
- Al-Refaie i in. 2021 — *TauREx 3: A Fast, Dynamic, and Extendable Framework for Retrievals*,
  ApJ 917, 37 ([arXiv:1912.07759](https://arxiv.org/abs/1912.07759),
  [doi:10.3847/1538-4357/ac0252](https://iopscience.iop.org/article/10.3847/1538-4357/ac0252))
- MacDonald 2023 — *POSEIDON: A Multidimensional Atmospheric Retrieval Code for Exoplanet Spectra*,
  JOSS 8(81), 4873 ([doi:10.21105/joss.04873](https://joss.theoj.org/papers/10.21105/joss.04873),
  [arXiv:2410.18181](https://arxiv.org/abs/2410.18181))
- Mollière i in. 2019 — *petitRADTRANS: a Python radiative transfer package for exoplanet
  characterization and retrieval*, A&A 627, A67 ([arXiv:1904.11504](https://arxiv.org/abs/1904.11504))

**Porównania i zgodność między kodami:**

- Barstow i in. 2022 — *A retrieval challenge exercise for the Ariel mission*
  ([arXiv:2203.00482](https://arxiv.org/abs/2203.00482)) — pięć kodów (ARCiS, NEMESIS, Pyrat Bay,
  TauREx 3, POSEIDON) na tym samym zbiorze syntetycznym
- MALBEC v1.0, 2024 — *Modeling Atmospheric Lines By the Exoplanet Community*, PSJ
  ([arXiv:2402.04329](https://arxiv.org/abs/2402.04329),
  [doi:10.3847/PSJ/ad2681](https://iopscience.iop.org/article/10.3847/PSJ/ad2681)) — intercomparison
  dziesięciu kodów transferu promienistego
- *Origin and extent of the opacity challenge for atmospheric retrievals of WASP-39 b*, 2023
  ([arXiv:2303.03383](https://arxiv.org/abs/2303.03383)) — wpływ wyboru modelu opacity na retrievowane
  abundancje
- Tennyson i in. 2024 — *The 2024 release of the ExoMol database*
  ([arXiv:2406.06347](https://arxiv.org/abs/2406.06347)) — baza linii molekularnych

**Benchmark i dane Ariel:**

- Aubin i in. 2023 — zwycięskie zgłoszenie ADC2023, normalizing flows
  ([arXiv:2309.09337](https://arxiv.org/abs/2309.09337))
- Mugnai i in. 2026 — *A public dataset of Ariel simulated observations*, RASTI
  ([arXiv:2605.03719](https://arxiv.org/abs/2605.03719)) — zbiór ADC2024: ExoSim2 + TauREx3,
  283 biny, szum instrumentalny, CIA włączone

**Metodologia SBI: kalibracja i misspecyfikacja:**

- Hermans i in. 2021 — *A Trust Crisis in Simulation-Based Inference? Your Posterior Approximations
  Can Be Unfaithful* ([arXiv:2110.06581](https://arxiv.org/abs/2110.06581))
- Ward i in. 2022 — *Robust Neural Posterior Estimation and Statistical Model Criticism* (RNPE),
  NeurIPS ([arXiv:2210.06564](https://arxiv.org/abs/2210.06564))
- Wehenkel i in. 2024 — *Addressing Misspecification in Simulation-based Inference through
  Data-driven Calibration* (RoPE) ([arXiv:2405.08719](https://arxiv.org/abs/2405.08719))
- Ruhlmann i in. 2025 — *Flow Matching Calibration for Simulation-Based Inference under Model
  Misspecification* (FMCPE) ([arXiv:2509.23385](https://arxiv.org/abs/2509.23385))

**Fizyka widm transmisyjnych i degeneracje**:

- Line & Parmentier 2016 ([arXiv:1511.09443](https://arxiv.org/abs/1511.09443)) — amplituda cech,
  degeneracja `T`–`μ`–chmury
- Heng & Kitzmann 2017 ([arXiv:1702.02051](https://arxiv.org/abs/1702.02051)) — wnioskowany jest
  `P_ref · X`, nie `X`
- Welbanks & Madhusudhan 2019 ([arXiv:1904.05356](https://arxiv.org/abs/1904.05356)) — degeneracja
  `R_p`–`P_ref` i granica licencji na jej zamrożenie
