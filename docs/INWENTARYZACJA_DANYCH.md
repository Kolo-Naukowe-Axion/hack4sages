# Inwentaryzacja danych — ExoBiome

Człon *inwentaryzacja* tematu umowy, w części dotyczącej danych. Uzupełnia
`docs/INWENTARYZACJA_MODELI.md` (modele i checkpointy).

**Zasada podziału wobec pozostałych dokumentów.** Ten dokument opisuje co mamy, w jakim
stanie, gdzie leży, czy da się użyć.

**Zasada dowodowa.** Każda komórka pochodzi z artefaktu, manifestu albo z payloadu checku audytowego,
nie z opisów z raportów zespołu. Gdzie liczba jest zmierzona w ramach tej inwentaryzacji, jest to napisane.

---

## 1. Tabela główna

| zbiór | ścieżka | generator + wersja | wiersze | rozmiar | stan | używany przez |
|---|---|---|---|---:|---|---|
| **ADC2023** | `data/ariel-ml-dataset/` | symulator organizatorów ADC2023 (dane zewnętrzne) | 41 423 train + 685 test | **1,9 GB** | **zdrowy** | model flagowy, model porównawczy NSF, baseline CNN |
| **crossgen** (tau + POSEIDON) | `data/TauREx set/` | TauREx **3.2.4** / POSEIDON **1.3.2** | 42 108 (41 423 tau + 685 POSEIDON) | 77 MB | **oba ramiona wadliwe** — patrz §4 | warianty `taurex_exobiome*`, `ariel_winner_on_taurex`, `taurex_fmpe` |
| **pRT validation** | `data/petitradtrans-adc2023-validation/` | petitRADTRANS **2.6.7** (+ taurex 3.2.4 do binowania) | 20 000 | **417 MB** | **nieużywany**; stanu nie weryfikował żaden check — §4 | żaden model, żaden skrypt, żaden check audytowy |
| **val_dataset** | `data/val_dataset/` | — (re-eksport ADC2023 do CSV) | 33 138 / 4 142 / 4 143 | 11 MB | **zdrowy, wtórny** | pomocniczo |
| **generated-data** | `data/generated-data/` | — (cache przygotowanych wejść) | `ariel_winner_nf_prepared` | 17 MB | scratch lokalny, ignorowany na `main` | `ariel_winner_nf` |

**Kod generatorów bez danych** (pakiety, które potrafią wygenerować zbiór, ale nie mają go obok siebie):

| pakiet | ścieżka | rozmiar | uwaga |
|---|---|---:|---|
| `crossgen_biosignatures` | `data/crossgen_biosignatures/` | 144 KB | generator zbioru **crossgen**; wejście do regeneracji (komenda w protokole §2) |
| `prt_adc2023_validation` | `data/prt_adc2023_validation/` | 72 KB | generator zbioru **pRT validation** |
| `prt_transmission_benchmark` | `data/prt_transmission_benchmark/` | 72 KB | generator benchmarku transmisyjnego; **odpowiadającego zbioru nie ma w repo** |

---

## 2. Gdzie te dane fizycznie są
Wyselekcjonowane zbiory i materiały EDA są celowo trzymane poza gałęzią główną, na
`origin/iwosmu/data-artifacts`, i podłączane osobnym worktree:

```bash
git fetch origin
git worktree add ../hack4sages-data origin/iwosmu/data-artifacts
```

Dane publikowane na tej gałęzi: `data/ariel-ml-dataset/`,
`data/petitradtrans-adc2023-validation/`, `data/reference_data/adc2023_reference_bundle.npz`,
`data/eda/`, `data/published/crossgen_biosignatures/20260311/`. Wyjścia generowane lokalnie mają
zostawać pod `data/generated-data/`. Na `main` ignorowane są zarówno one, jak i podłączane zbiory:
`.gitignore:23,25,40` (`data/generated-data/`, `data/val_dataset/`, `data/ariel-ml-dataset/`).

---

## 3. Krótka charakterystyka użytych generatorów

### 3.1 parametry crossgen

Oba ramiona crossgen pochodzą z **jednego** przebiegu, z tych samych rozkładów priorów
(`data/crossgen_biosignatures/constants.py:28-37`), `MASTER_SEED = 20260310`:

| parametr | zakres |
|---|---|
| promień planety | 0,7 – 1,5 R_jup |
| `log g` | 2,8 – 3,7 (cgs) |
| temperatura | 500 – 1800 K |
| promień gwiazdy | 0,2 – 1,3 R_sun |
| log₁₀ VMR (5 gazów) | −12 – −2, przy `Σ X_i ≤ 0,10` |
| próg obecności gazu | log₁₀ VMR = **−8,0** |
| tło | H₂ 0,85 / He 0,15 — **deklarowane w konfiguracji** (`constants.py:35-36`); zrealizowane VMR są niższe, bo gazy śladowe zabierają do `TRACE_VMR_MAX_TOTAL = 0,10`, patrz K11/`a21` |

Szum: `iid_gaussian_white`, **skalar na próbkę** (nie per bin), σ 20,0 – 100,0 ppm
(`manifest.json: noise_model`).

### 3.2 Czym nasze wygenerowane TauREx i POSEIDON się różnią
**NOTKA**: to jest nasza konfiguracja generacji danych, stąd właśnie wynikają różnice (jak np. ta w punkcie 1 tabeli).

| własność | TauREx 3.2.4 | POSEIDON 1.3.2 | dowód |
|---|---|---|---|
| wkłady do opacity | `AbsorptionContribution` + `RayleighContribution` — **2 elementy**, brak CIA | absorpcja + Rayleigh + **CIA** — biblioteka alokuje kanał CIA sama, bez naszego udziału | **tau:** `taurex_backend.py:148,154-157` (jedyne wywołanie w pliku) — lektura wykonana na `taurex 3.3.2` z `.venv-cnn` **POSEIDON:** dowód na wersji generującej **1.3.2** — `X_CIA` w `atmosphere_keys` (`reports/audit/d01_poseidon_132/d01_poseidon_diagnosis.json`) |
| rozdzielczość natywna | R = 100 (bez rebinowania) | **R = 1000**, rebinowana do 100 | `meta/tau_generation.json`, `meta/poseidon_generation.json` (`poseidon_native_resolution`) |
| liczba shardów przy generacji | 162 × 256 wierszy | 1 × 685 | `meta/*_generation.json` |
| profil T–P | izoterma | izoterma (`PT_profile="isotherm"`) | `poseidon_backend.py:108` |
| siatka ciśnień | 100 poziomów, 1e-6 – **100 bar** | ta sama | `constants.py:39-41` |
| tło H₂/He | 0,85 / 0,15 podane wprost | ta sama proporcja, przez `He_fraction = He/H2 = 0,17647` | `constants.py:35-37`, `poseidon_backend.py:171`; **zrealizowane** `vmr_h2 ∈ [0,8275; 0,8500]`, `vmr_he ∈ [0,1460; 0,1500]` — `a21` |
| baza opacity | wczytana | **`opacity_files = []`, `input_data_root = null`** | `d01` stage 1 |
| siatka docelowa | 218 binów, R = 100, 0,6 – 5,25 µm | **ta sama** (wspólna dla obu ramion) | `manifest.json: wavelength_grid` |

Dwie uwagi, obie istotne przy interpretacji:

1. **Siatka ciśnień to 100 bar, nie domyślne 10 bar TauREx-a.** To wpływa na definicję ciśnienia
   referencyjnego, a więc na przelicznik między błędem temperatury i błędem abundancji.
2. **Różnica „TauREx vs POSEIDON" w tym repo nie jest różnicą dwóch poprawnych modeli fizycznych**, bo
   jedna strona nie ma kontinuum, druga nie ma bazy przekrojów czynnych.

### 3.3 Generator pRT validation

petitRADTRANS 2.6.7, opacity R = 400, 11 gatunków linii
(`H2O_HITEMP`, `CO_all_iso_HITEMP`, `CH4`, `NH3`, `CO2`, `H2S`, `VO`, `TiO_all_Exomol`, `PH3`,
`Na_allard`, `K_allard`) **plus chmury** (`MgSiO3(c)_cd`, `Fe(c)_cd`). Binowanie do siatki ADC2023 przez `taurex` 3.2.4 `FluxBinner`. Widma zapisane przeskalowane
(`scale_factor = 1e16`), σ w jednostkach `1e-16 W/m²/µm`, zakres 0,05 – 0,50.

---

## 4. Stan z dowodem

| zbiór | stan | dowód |
|---|---|---|
| **ADC2023** | **zdrowy** | dane pobrane od organizatorów challenge'u, nie generowane u nas|
| **crossgen tau** | **niepełny fizycznie** — generator wywołany **bez CIA**, głównego kontinuum w atmosferze H₂/He | `taurex_backend.py:154-157`: `contributions` ma 2 elementy, brak `CIAContribution`; oficjalny forward model organizatorów ma 3 (`FM_utils_final.py:207-209`) — ustalenie **K11** |
| **crossgen POSEIDON** | **zepsuty** — 685/685 widm `transit_depth_noiseless` ma **jedną wartość na 218 binów**; pole czytane przez modele (`transit_depth_noisy`) ma `amp/σ = 0,998` i `SNR > 3` w **zerowej** liczbie wierszy | `a01`, kryterium bezprogowe; przyczyna zawężona do nasyconej ekstynkcji przez `d01` stage 1 — **K1**, **K1(c)** |
| **pRT validation** | **nieoceniony** — zbiór nie jest używany | fakt z manifestu (`manifest.json`): widma zapisane przeskalowane `scale_factor = 1e16`, σ podana osobno w jednostkach `1e-16 W/m²/µm`, zakres `0,05 – 0,50`.|
| **val_dataset** | **zdrowy, ale wtórny** | `manifest.json`: `split_seed = 42`, frakcje 0,8/0,1/0,1, wiersze 33 138 / 4 142 / 4 143 — identyczne z `split_manifest.json` modelu flagowego; to re-eksport ADC2023 do CSV, nie niezależne dane |

---

## 5. Etykiety i cele

| zbiór | plik etykiet | cele | uwaga |
|---|---|---|---|
| ADC2023 | `TrainingData/Ground Truth Package/FM_Parameter_Table.csv` | **7**: `planet_radius`, `planet_temp` + 5 log-abundancji | dostępny też `Tracedata.hdf5` (posterior referencyjny nested sampling) i `QuartilesTable.csv` |
| crossgen | `labels.parquet` (+ `latents.parquet`) | 5 log₁₀ VMR, obecność per gaz, `temperature_k`, `log_g_cgs`, promienie | `temperature_k` **jest w etykietach**, ale nie jest podawana modelom jako cecha — ustalenie **K9** |
| pRT validation | `ValidationData/Ground Truth Package/FM_Parameter_Table.csv` | jak ADC2023 | plus `NativeSpectra_R400.hdf5` (widma przed binowaniem) |

`Tracedata.hdf5` z ADC2023 jest jedynym w repo **posteriorem referencyjnym**. To na nim stoją pomiary
sprzężenia temperatura↔abundancja z **K9** (663 planety mające pełną 7-kolumnową macierz trace, czyli
16 % holdoutu; dostępność trace'u jest skorelowana z numerem planety, patrz zastrzeżenie w K9).

---

## 6. Co z tego wynika dla odbioru

1. **Trzy niezależne zbiory widm**, z których **jeden jest zdrowy** ADC2023, i to on nosi wszystkie
   liczby, które wolno cytować jako wynik modelu.
2. **Oś cross-generator nie ma zdrowej strony.** jedna strona nie ma
   kontinuum CIA, druga nie ma wczytanej bazy opacity.
3. **Zbiór pRT jest w repo i nie jest przez nic używane.**
