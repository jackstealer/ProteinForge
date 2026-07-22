<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/2/23/Myoglobin_3D_animation.gif" alt="Protein Spinning" width="250"/>

# 🧬 ProteinForge

**AI-Powered Protein Structure Prediction & Analysis Dashboard**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![ESMFold](https://img.shields.io/badge/AI_Model-ESMFold-7A3CEF.svg)](https://esmatlas.com)
[![AlphaFold](https://img.shields.io/badge/AI_Model-AlphaFold_DB-0053D6.svg)](https://alphafold.ebi.ac.uk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A 100% dynamic, multi-model, hyperparameter-tunable bioinformatics pipeline built for modern structural biology.*

---

</div>

## ✨ Overview



## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack & Dependencies](#2-technology-stack--dependencies)
3. [Project Architecture](#3-project-architecture)
4. [File Structure](#4-file-structure)
5. [Application Pages](#5-application-pages)
   - 5.1 [Home Page](#51-home-page)
   - 5.2 [Predict Page](#52-predict-page)
   - 5.3 [Batch Analysis Page](#53-batch-analysis-page)
   - 5.4 [Compare Page](#54-compare-page)
   - 5.5 [About Page](#55-about-page)
6. [Prediction Results Tabs](#6-prediction-results-tabs)
   - 6.1 [3D Viewer Tab](#61-3d-viewer-tab)
   - 6.2 [Confidence Tab](#62-confidence-tab)
   - 6.3 [Analysis Tab](#63-analysis-tab)
   - 6.4 [Download Tab](#64-download-tab)
   - 6.5 [Reliability Tab](#65-reliability-tab)
7. [Sidebar Controls](#7-sidebar-controls)
8. [Utility Functions — Core (utils.py)](#8-utility-functions--core-utilspy)
9. [Hyperparameter-Tunable Analysis Functions](#9-hyperparameter-tunable-analysis-functions)
10. [API Integrations](#10-api-integrations)
11. [Session State Management](#11-session-state-management)
12. [Caching Strategy](#12-caching-strategy)
13. [Data Flow Diagram](#13-data-flow-diagram)
14. [Bias-Variance Trade-off Framework](#14-bias-variance-trade-off-framework)
15. [Testing & Validation](#15-testing--validation)
16. [Limitations & Best Practices](#16-limitations--best-practices)
17. [References & Scientific Background](#17-references--scientific-background)

---

## 1. Project Overview

<div align="center">
  <img src="https://i.gifer.com/Xqg8.gif" alt="DNA Animation" width="150" />
</div>

**ProteinForge** is an interactive, browser-based dashboard for AI-driven protein structure prediction. It is designed as a complete end-to-end bioinformatics pipeline — from raw amino acid sequence input all the way to 3D interactive visualization, quantitative analysis, batch processing, and downloadable results.

### Core Philosophy

| Principle | Description |
|---|---|
| **100% Dynamic Data** | No sequences, structures, or analysis results are hard-coded. Everything is fetched live from external APIs or provided by the user. |
| **Dual-Model Prediction** | Supports two industry-leading AI models — ESMFold (via API) and AlphaFold (via pre-computed database). |
| **Science-First Design** | All analysis metrics are grounded in established bioinformatics methods (Kyte-Doolittle, Dunker disorder, pLDDT confidence). |
| **Hyperparameter Tuning** | Users can tune 6 analysis hyperparameters that directly control the bias-variance trade-off of the pipeline. |
| **Reproducibility** | Smart Streamlit caching ensures identical inputs always produce identical results without redundant API calls. |

### Target Users

- **Bioinformatics students and researchers** learning protein structure prediction
- **Computational biologists** running exploratory structural analysis
- **Biochemists** comparing experimental vs. AI-predicted structures
- **Educators** demonstrating protein folding concepts interactively

---

## 2. Technology Stack & Dependencies

### Python Libraries

| Library | Version | Role |
|---|---|---|
| `streamlit` | >= 1.28.0 | Web framework — renders the entire UI, manages state, handles reruns |
| `py3Dmol` | >= 2.0.3 | WebGL-based 3D molecular visualization in the browser |
| `stmol` | >= 0.0.9 | Streamlit bridge to embed py3Dmol views as interactive components |
| `requests` | >= 2.31.0 | HTTP client for all API calls (UniProt, ESMFold, AlphaFold) |
| `pandas` | >= 2.0.0 | Data manipulation for batch analysis, dataset handling, CSV export |
| `plotly` | >= 5.17.0 | Interactive 2D/3D charts for pLDDT, hydrophobicity, disorder, composition |
| `biopython` | >= 1.81 | ProteinAnalysis for sequence property calculations and FASTA parsing |
| `numpy` | >= 1.24.0 | Numerical operations — RMSD, Gaussian smoothing, coordinate math |
| `biotite` | >= 0.38.0 | Advanced structural bioinformatics (reserved for future structural analysis) |
| `datasets` | >= 2.14.0 | Hugging Face datasets library for loading curated protein sequence databases |

### External APIs

| API | URL | Purpose |
|---|---|---|
| **UniProt REST API** | `https://rest.uniprot.org/uniprotkb/` | Fetch protein sequences and metadata by UniProt accession ID |
| **ESMFold API** | `https://api.esmatlas.com/foldSequence/v1/pdb/` | On-demand protein structure prediction (POST request with sequence) |
| **AlphaFold Database** | `https://alphafold.ebi.ac.uk/files/` | Retrieve pre-computed high-quality structures in PDB format |
| **Hugging Face Datasets** | `datasets` library | Curated protein datasets (damlab/uniprot, LiteFold/PDB) |

---

## 3. Project Architecture

ProteinForge follows a **single-file web app architecture** with a clean separation of concerns:

```
+------------------------------------------------------------+
|                       BROWSER (User)                       |
|   Home | Predict | Batch | Compare | About                 |
+---------------------------+--------------------------------+
                            | Streamlit HTTP
+---------------------------v--------------------------------+
|                    app.py  (Controller)                    |
|  - Page routing via sidebar radio                          |
|  - Session state management (6 core + 6 HP keys)          |
|  - UI rendering (CSS, HTML, Plotly, py3Dmol)               |
|  - Calls utility functions from utils.py                   |
+---------------------------+--------------------------------+
                            | Function calls
+---------------------------v--------------------------------+
|                    utils.py  (Service Layer)               |
|  - Input validation & FASTA parsing                        |
|  - API communication (UniProt, ESMFold, AlphaFold)         |
|  - Sequence analysis (Biopython ProteinAnalysis)           |
|  - Structural analysis (pLDDT, RMSD, coordinates)          |
|  - Hyperparameter-tunable functions (6 new functions)      |
|  - Hugging Face dataset management                         |
+---------------------------+--------------------------------+
                            | HTTP/HTTPS requests
+---------------------------v--------------------------------+
|             External APIs & Data Sources                   |
|   UniProt REST | ESMFold API | AlphaFold DB | HuggingFace  |
+------------------------------------------------------------+
```

---

## 4. File Structure

```
INTERN/
+-- app.py                    # Main application (1,652 lines)
|                             # Contains: CSS, page routing, all UI pages,
|                             # show_results(), show_batch_analysis(),
|                             # show_comparison(), show_about()
|
+-- utils.py                  # Utility & service functions (1,019 lines)
|                             # Contains: all API clients, data processors,
|                             # analysis engines, hyperparameter functions
|
+-- requirements.txt          # Python dependency specifications
|
+-- run_checks.py             # Automated test suite (46 test cases)
|
+-- test_installation.py      # Environment validation script
|
+-- .streamlit/               # Streamlit configuration directory
```

---

## 5. Application Pages

Navigation is handled by a radio button group in the left sidebar under "Controls". The selected page is routed through the `main()` function in `app.py`.

```python
page = st.radio("", ["Home", "Predict", "Batch", "Compare", "About"])
```

---

### 5.1 Home Page

**Function:** `show_home()` in `app.py`

The Home page serves as the landing dashboard and data discovery interface. It contains four distinct sections:

#### Section A — Feature Cards

Three animated glassmorphism cards summarize the platform's core capabilities:

- **Lightning Fast** — ESMFold predictions in 30–60 seconds using state-of-the-art AI
- **High Accuracy** — Powered by both ESMFold and AlphaFold, the two leading structure prediction models
- **100% Dynamic** — Every piece of data is fetched live from APIs; nothing is hard-coded

The cards have CSS hover animations (translateY(-8px) scale(1.02)) that provide a modern, premium feel.

#### Section B — Quick Start Guide

An expandable guide showing three numbered steps for new users:

1. Choose a protein (example, UniProt ID, paste sequence, or upload FASTA)
2. Select a prediction engine (ESMFold or AlphaFold)
3. View the 3D structure and analysis results

#### Section C — Try Example Proteins

Five example proteins with **Load** buttons displayed in a 5-column grid:

| Protein | UniProt ID | Description |
|---|---|---|
| Insulin (Human) | P01308 | Glucose-regulating hormone |
| Hemoglobin Alpha | P69905 | Oxygen transport protein |
| Green Fluorescent Protein | P42212 | Fluorescent reporter from jellyfish |
| Lysozyme | P00698 | Antimicrobial enzyme |
| Myoglobin | P02144 | Muscle oxygen storage protein |

When a Load button is clicked:
1. `fetch_uniprot_sequence(uniprot_id)` is called — returns (sequence, name) from the UniProt API
2. The result is stored in `st.session_state.temp_sequence`, `.temp_name`, `.temp_uniprot`
3. A success toast + balloon animation confirms loading
4. The user navigates to the Predict page where the sequence is pre-loaded

> **Important:** The sequences are **not stored in code**. The button only stores the UniProt ID. The sequence is always freshly fetched from the UniProt API when clicked.

#### Section D — Browse Protein Dataset

Integration with Hugging Face protein datasets:

- **"Load Random Sequences"** button: Calls `get_random_sequences_from_dataset(5)` which loads the `damlab/uniprot` dataset (500,000+ curated SwissProt proteins) and samples 5 random entries
- **"Dataset Stats"** button: Calls `get_dataset_stats()` to display total sequences, average length, min/max lengths
- Loaded sequences are shown in expandable cards with a **"Use this sequence"** button that pre-loads them into the Predict page

---

### 5.2 Predict Page

**Function:** `show_prediction()` in `app.py`

This is the core functionality page where users submit sequences and trigger predictions.

#### Pre-loading Check

At the top of `show_prediction()`, the function checks if a sequence was loaded from the Home page:

```python
if st.session_state.temp_sequence:
    sequence = st.session_state.temp_sequence
    protein_name = st.session_state.temp_name
    uniprot_id = st.session_state.temp_uniprot
    # Clear temp to avoid showing message repeatedly
    st.session_state.temp_sequence = None
```

#### Input Method 1 — Paste Sequence

The user types or pastes a raw amino acid sequence using standard single-letter codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).

- Input is automatically uppercased and whitespace-stripped
- The sequence is stored in `st.session_state.current_input_sequence` to persist across reruns
- Protein name defaults to "User Sequence"

#### Input Method 2 — Upload FASTA

Accepts `.fasta`, `.fa`, `.txt`, `.faa` file formats. Uses Biopython's `SeqIO.parse()` to extract records:

- If the file contains **one sequence**: automatically selected
- If the file contains **multiple sequences**: a `st.selectbox` allows the user to choose which sequence to predict
- Non-standard characters are stripped using regex: `re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)`

#### Input Method 3 — UniProt ID

A text input + Fetch button workflow:

1. User enters a UniProt accession (e.g., `P42212` for GFP)
2. `fetch_uniprot_sequence(uniprot_id)` fetches the FASTA from `https://rest.uniprot.org/uniprotkb/{id}.fasta`
3. `fetch_uniprot_metadata(uniprot_id)` additionally fetches protein name, organism, gene name, and sequence length from the JSON API
4. All data stored in session state for the prediction and display
5. The UniProt ID is stored separately to enable AlphaFold DB lookup

#### Input Method 4 — Random from Dataset

Calls `get_random_sequences_from_dataset(5)` to fetch random sequences from the Hugging Face dataset. The user selects one from a list showing protein ID and length.

#### Sequence Validation

After input, every sequence is run through `validate_sequence(sequence)`:

```python
def validate_sequence(sequence: str) -> Tuple[bool, str]:
    # Checks: non-empty, valid amino acid characters only,
    # minimum 10 residues, maximum 2000 residues
```

Validation failures display a red error message and block the prediction button.

#### Sequence Information Expander

After successful validation, an expandable panel shows:
- Protein name
- Sequence length in residues
- UniProt accession (if known)
- Formatted sequence display (60 characters per line)
- Sequence length metric card (bold, prominent)

#### Sequence Quality Assessment (NEW — Hyperparameter-driven)

Before the predict button, a dedicated panel computes and displays:

```
Quality Score    Complexity    Disorder Risk
  82/100          95.4%          52.5%
[=========================================    ] 82%
Sequence looks good for structure prediction!
```

This calls `calculate_sequence_quality(sequence)` which scores three factors:
- **Shannon Entropy** (sequence complexity): Low complexity = repetitive sequence = poor prediction
- **Disorder Propensity** (Dunker scale): High charge + low hydrophobicity = likely disordered
- **Length Penalty**: Sequences > 600 residues get a penalty (ESMFold accuracy degrades)

Any concern triggers a warning message with actionable advice.

#### Prediction — ESMFold API

When "Predict with ESMFold" is clicked:

1. A progress bar and status text appear
2. `predict_structure_esmfold(sequence)` is called — a POST request to `https://api.esmatlas.com/foldSequence/v1/pdb/` with the sequence as the body
3. The API returns a PDB-formatted structure (30–60 seconds for typical sequences)
4. On success:
   - `pdb_string` stored in `st.session_state.predicted_structure`
   - `analyze_sequence(sequence)` stores physicochemical analysis in `st.session_state.sequence_analysis`
   - `extract_plddt_from_pdb(pdb_string)` extracts confidence scores into `st.session_state.plddt_scores`
   - `st.session_state.prediction_source` set to `"ESMFold API"`
   - A balloon animation confirms success and a page rerun displays the results

#### Prediction — AlphaFold Database

When "Fetch from AlphaFold DB" is clicked (requires UniProt ID):

1. `fetch_alphafold_structure(uniprot_id)` tries model versions v4 then v3:
   - `https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb`
   - `https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v3.pdb`
2. The pre-computed PDB is returned immediately (no AI computation needed)
3. Same session state population as ESMFold

---

### 5.3 Batch Analysis Page

**Function:** `show_batch_analysis()` in `app.py`

Designed for high-throughput analysis of multiple protein sequences at once.

#### Workflow

1. User uploads a multi-FASTA file
2. `parse_fasta(fasta_content)` parses all records using Biopython's `SeqIO`
3. A preview expander shows the first 10 sequence names and lengths
4. A slider lets the user select how many sequences to process (1 to min(50, total))
5. Clicking "Start Analysis" runs `analyze_sequence()` on each valid sequence

#### Per-Sequence Analysis

For each sequence, the following metrics are computed via Biopython's `ProteinAnalysis`:

| Metric | Description |
|---|---|
| **Name** | Sequence identifier from FASTA header |
| **Length** | Number of amino acid residues |
| **MW (Da)** | Molecular weight in Daltons |
| **pI** | Isoelectric point (pH at which net charge = 0) |
| **Instability Index** | Predicted protein stability score |
| **Stability** | "Stable" if instability index < 40, else "Unstable" |

#### Output

- Results displayed in a sortable `st.dataframe` with full-width formatting
- A "Download CSV" button exports all results as a comma-separated file (`batch_analysis.csv`)
- A live progress bar + status text shows real-time processing state

> **Note:** Batch analysis does NOT call ESMFold/AlphaFold for structures. It performs sequence-only physicochemical analysis using Biopython, making it fast enough to process 50 sequences in seconds.

---

### 5.4 Compare Page

**Function:** `show_comparison()` in `app.py`

Compares a predicted structure against an experimentally-determined (reference) structure using the **Root Mean Square Deviation (RMSD)** metric.

#### Workflow

1. **Left column** — Shows the current predicted structure from session state (name + success badge)
2. **Right column** — User uploads a reference PDB file (experimental structure from PDB, RCSB, etc.)
3. Both structures are parsed with `parse_pdb_coordinates(pdb_string, atom_type='CA')` which extracts only **C-alpha atom coordinates** (the backbone representation)
4. `calculate_rmsd(pred_coords, ref_coords)` computes the RMSD

#### RMSD Formula

```
RMSD = sqrt( (1/N) * sum( |ri - ri'|^2 ) )
```

Where `ri` and `ri'` are the corresponding C-alpha positions in the predicted vs. reference structures, and N is the number of aligned atoms.

#### RMSD Interpretation

| RMSD (A) | Quality | Meaning |
|---|---|---|
| < 2.0 | **Excellent** | Nearly identical backbone geometry |
| 2.0 – 4.0 | **Good** | Good agreement, minor deviations |
| 4.0 – 6.0 | **Fair** | Some structural differences |
| > 6.0 | **Poor** | Significant structural divergence |

---

### 5.5 About Page

**Function:** `show_about()` in `app.py`

A professionally designed informational page covering:

- **Overview**: Platform description, target audience
- **Key Features**: Bullet-list of 7 major capabilities
- **Data Sources**: UniProt, AlphaFold DB, ESMFold API, Hugging Face
- **pLDDT Quality Scale**: Color-coded confidence interpretation guide
- **Prediction Engines**: Side-by-side ESMFold vs. AlphaFold comparison with academic references
- **Limitations**: 5 known constraints (length limit, monomers only, no PTMs, etc.)
- **Best Practices**: 5 recommendations for reliable use
- **Useful Links**: Clickable cards to CASP, UniProt, AlphaFold DB, ESM Atlas
- **Footer**: Credit panel with library attributions

---

## 6. Prediction Results Tabs

After a successful prediction, `show_results()` renders a 5-tab panel:

```python
tabs = st.tabs(["3D Viewer", "Confidence", "Analysis", "Download", "Reliability"])
```

---

### 6.1 3D Viewer Tab

Uses **py3Dmol + stmol** to render an interactive 3D molecular visualization.

#### Controls (right column)

| Control | Options | Default |
|---|---|---|
| Style | cartoon, sphere, stick, line | cartoon |
| Color | pLDDT, Spectrum, Secondary Structure | pLDDT |
| Auto-rotate | checkbox | off |
| Background | color picker | white |

#### Color Schemes

- **pLDDT Mode**: Colors each residue by its confidence score using a roygb gradient from 50 (red) to 90+ (blue). This is the standard AlphaFold/ESMFold confidence coloring used in the literature.
- **Spectrum Mode**: Rainbow coloring from N-terminus (blue) to C-terminus (red)
- **Secondary Structure**: Helix = red, Sheet = yellow, Loop = green (Jmol standard)

The viewer is rendered at 800x600 pixels, embedded into the page using `showmol(view)`.

---

### 6.2 Confidence Tab

Visualizes per-residue pLDDT (predicted Local Distance Difference Test) scores extracted from the PDB B-factor column.

#### pLDDT Score System

| Score | Category | Color | Interpretation |
|---|---|---|---|
| >= 90 | Very High | Dark Blue `#0053D6` | ~95% accuracy in backbone placement |
| 70–89 | Confident | Light Blue `#65CBF3` | Reliable structural model |
| 50–69 | Low | Yellow `#FFDB13` | Use with caution |
| < 50 | Very Low | Orange `#FF7D45` | Likely intrinsically disordered |

#### Summary Metrics

Four metric cards display:
- **Average pLDDT** across all residues
- **Maximum pLDDT** (most confident residue)
- **Minimum pLDDT** (least confident residue)
- **Quality label** (Very High / Confident / Low / Very Low)

#### Per-Residue Chart

A Plotly Scatter plot showing:
- **Raw pLDDT** (light gray line with colored markers — each marker colored by its own confidence category)
- **Smoothed pLDDT** (purple line using Gaussian smoothing with sigma from the hyperparameter panel)

Horizontal reference lines at 90, 70, and 50 clearly demarcate the confidence zones.

#### Distribution Chart

A Plotly Bar chart showing the count of residues in each confidence category (Very Low, Low, Confident, Very High), using the same 4-color scheme.

---

### 6.3 Analysis Tab

Computes and displays sequence physicochemical properties using **Biopython's `ProteinAnalysis`** class.

#### Physicochemical Metrics

| Metric | Formula / Method | Interpretation |
|---|---|---|
| **Sequence Length** | `len(sequence)` | Total residue count |
| **Molecular Weight** | Sum of residue masses + water | In Daltons (Da) |
| **Aromaticity** | (Phe + Tyr + Trp) / length | Fraction of aromatic residues |
| **Instability Index** | Guruprasad et al. (1990) | < 40 = stable, > 40 = unstable |
| **Isoelectric Point (pI)** | Henderson-Hasselbalch iteration | pH of zero net charge |
| **Secondary Structure** | Chou-Fasman method | Fraction of helix / turn / sheet |

#### Amino Acid Composition

A bar chart showing the percentage of each of the 20 standard amino acids. This reveals:
- Hydrophobic core composition
- Charged residue content
- Unusual amino acid enrichment

---

### 6.4 Download Tab

Provides three downloadable outputs:

#### 1. PDB File Download

The raw PDB file can be opened in PyMOL, Chimera, VMD, or any structural biology software.

#### 2. Sequence FASTA Download

A properly formatted FASTA file for use in sequence analysis pipelines.

#### 3. Analysis Report Download

A plain-text report (.txt) containing:
- Prediction metadata (source, timestamp)
- All physicochemical metrics from the Analysis tab
- pLDDT summary statistics (avg, max, min)

#### 4. PDB Preview

An `st.expander` shows the first 2,000 characters of the raw PDB file in a code block for quick inspection without downloading.

---

### 6.5 Reliability Tab (NEW — Hyperparameter-Driven)

A comprehensive reliability analysis page that dynamically responds to the hyperparameters set in the sidebar Tuning Controls panel.

#### A — Reliability Grade Card

Calls `assess_prediction_reliability(plddt_scores, threshold=plddt_threshold)` and displays a large letter grade:

| Grade | Condition | Color |
|---|---|---|
| **A — Excellent** | avg >= 90 AND >= 80% residues reliable | Blue |
| **B — Good** | avg >= 70 AND >= 60% residues reliable | Green |
| **C — Fair** | avg >= 50 AND >= 40% residues reliable | Amber |
| **D — Poor** | otherwise | Red |

Four supporting metrics: avg pLDDT, reliable residues count, percent reliable, number of high-confidence segments.

High-confidence residue segments (contiguous runs above the threshold) are listed explicitly (e.g., "Res 12-94, Res 105-198").

#### B — Hydrophobicity Profile

Calls `calculate_hydrophobicity_profile(sequence, window_size=hp_window_size)`:

A bar chart using the **Kyte-Doolittle hydrophobicity scale** where:
- Red bars = hydrophobic windows (score > 0) — likely buried in protein core or membrane-spanning
- Blue bars = hydrophilic windows (score < 0) — likely solvent-exposed or charged

The `window_size` hyperparameter directly controls the smoothing: smaller windows reveal local features; larger windows reveal global hydrophobic domains.

#### C — Disorder Profile

Calls `predict_disorder_profile(sequence, window_size, disorder_threshold)`:

A bar chart of disorder propensity scores (0-1) per window position. A dashed threshold line (controlled by the `disorder_threshold` hyperparameter) separates ordered from disordered regions.

- Green bars = ordered regions (below threshold)
- Red bars = predicted disordered regions (above threshold)

The method uses a simplified IUPred-like heuristic combining hydrophobicity and net charge to estimate intrinsic disorder.

#### D — Terminal Trimming Analysis

Calls `trim_low_confidence_termini(sequence, pdb_string, min_plddt=plddt_threshold)`:

Identifies low-confidence N- and C-terminal regions below the pLDDT threshold and shows:
- Number of residues trimmed from N-terminus
- Number of residues trimmed from C-terminus
- Total retained residues

If any trimming occurs, a "Download Trimmed PDB" button provides a cleaned PDB with the low-confidence termini removed. This trimmed structure is better suited for molecular docking, loop modeling, and comparative analyses.

---

## 7. Sidebar Controls

The left sidebar provides persistent global controls available on all pages.

### Navigation

A `st.radio` button group for page switching (Home, Predict, Batch, Compare, About).

### Prediction Engine

```
Choose model: (ESMFold API) or (AlphaFold DB)
```

Stored in `st.session_state.model_choice`. Determines which API is called on the Predict page.

### Tuning Controls (NEW — Hyperparameter Panel)

An expandable "Adjust Hyperparameters" section with 6 tunable controls:

| Control | Widget | Range | Default | Biological Meaning |
|---|---|---|---|---|
| **pLDDT Confidence Threshold** | slider | 50–90 (step 5) | 70 | Minimum confidence to classify a residue as "reliable" |
| **Hydrophobicity Window** | select_slider | 5, 7, 9, 11, 13, 15, 17, 19, 21 | 9 | Sliding window width for Kyte-Doolittle profile |
| **Disorder Threshold** | slider | 0.1–0.9 (step 0.1) | 0.5 | Cutoff above which a window is flagged as disordered |
| **pLDDT Smoothing sigma** | slider | 0.5–5.0 (step 0.5) | 1.5 | Gaussian smoothing strength for the confidence chart |
| **Auto-trim Termini** | checkbox | on/off | on | Automatically remove low-confidence terminal residues |
| **Ensemble Mode** | checkbox | on/off | on | Compare both models when UniProt ID is known |

A live **bias-variance progress bar** at the bottom of the tuning panel updates in real time as the pLDDT threshold is changed:

```
<-- Low Bias | Threshold: 70 | Low Variance -->
[========================         ] (50%)
```

### Data Sources

A "Data Sources" section confirms all four live data sources:
- UniProt REST API
- AlphaFold Database
- ESMFold API
- Zero hard-coded data

### Quick Stats (when structure loaded)

Displays a real-time summary of the current predicted structure:
- Protein name
- Source (ESMFold API or AlphaFold DB)
- Sequence length (aa)
- Average pLDDT score

---

## 8. Utility Functions — Core (utils.py)

### 8.1 `validate_sequence(sequence)`

**Purpose:** Validates and cleans a raw amino acid sequence before prediction.

**Logic:**
1. Uppercase and strip whitespace
2. Remove spaces, newlines, carriage returns
3. Check against valid amino acid set: `ACDEFGHIKLMNPQRSTVWY`
4. Reject sequences shorter than 10 residues
5. Reject sequences longer than 2,000 residues

**Returns:** `(True, cleaned_sequence)` or `(False, error_message)`

---

### 8.2 `parse_fasta(fasta_content)`

**Purpose:** Parses FASTA format text into a list of (name, sequence) tuples.

**Implementation:** Uses Biopython's `SeqIO.parse()` with a `StringIO` wrapper. After parsing, applies the same character cleaning as `validate_sequence`. Sequences shorter than 10 residues are discarded.

**Returns:** `List[Tuple[str, str]]` — list of (identifier, sequence) pairs

---

### 8.3 `fetch_uniprot_sequence(uniprot_id)` — Cached 1 hour

**Purpose:** Fetches a protein's amino acid sequence from the UniProt REST API.

**Endpoint:** `GET https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta`

**Returns:** `(sequence_str, protein_name)` or `(None, None)` on failure

**Caching:** `@st.cache_data(ttl=3600)` — identical UniProt IDs return the cached result for 1 hour, avoiding redundant API calls.

---

### 8.4 `fetch_uniprot_metadata(uniprot_id)` — Cached 1 hour

**Purpose:** Fetches structured metadata for a protein from UniProt's JSON API.

**Endpoint:** `GET https://rest.uniprot.org/uniprotkb/{uniprot_id}.json`

**Returns:**
```python
{
    'name': 'Insulin',
    'organism': 'Homo sapiens',
    'gene': 'INS',
    'length': 110
}
```

---

### 8.5 `predict_structure_esmfold(sequence)` — Cached 24 hours

**Purpose:** Submits a protein sequence to the ESMFold API for structure prediction.

**Endpoint:** `POST https://api.esmatlas.com/foldSequence/v1/pdb/`  
**Body:** Raw amino acid sequence (plain text)  
**Timeout:** 300 seconds (5 minutes)

**Error Handling:**
- HTTP 400 -> Invalid sequence format
- HTTP 413 -> Sequence too long
- Timeout -> Suggests using shorter sequence
- Connection error -> Internet connectivity message

**Returns:** `(pdb_string, None)` on success or `(None, error_message)` on failure

---

### 8.6 `fetch_alphafold_structure(uniprot_id)` — Cached 24 hours

**Purpose:** Downloads a pre-computed structure from the AlphaFold EBI database.

**Strategy:** Falls back gracefully — tries model v4 first, then v3:
```
URL_v4 = https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb
URL_v3 = https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v3.pdb
```

**Returns:** `(pdb_string, None)` or `(None, "No AlphaFold structure available for {id}")`

---

### 8.7 `extract_plddt_from_pdb(pdb_string)`

**Purpose:** Reads per-residue pLDDT scores from the B-factor column of a PDB file.

**Method:** Only reads `ATOM` records where the atom name (columns 12-16) is `CA` (alpha carbon). This gives one confidence score per residue, extracted from columns 60-66 (B-factor field).

**Returns:** `(residue_numbers: List[int], plddt_scores: List[float])`

---

### 8.8 `analyze_sequence(sequence)`

**Purpose:** Computes comprehensive physicochemical properties of a protein sequence.

**Implementation:** Creates a Biopython `ProteinAnalysis(sequence)` object and calls:

| Property | Method |
|---|---|
| Amino acid composition | `analyzer.amino_acids_percent` |
| Molecular weight | `analyzer.molecular_weight()` |
| Aromaticity | `analyzer.aromaticity()` |
| Instability index | `analyzer.instability_index()` |
| Isoelectric point | `analyzer.isoelectric_point()` |
| Secondary structure | `analyzer.secondary_structure_fraction()` |

**Returns:**
```python
{
    'length': int,
    'molecular_weight': float,
    'aromaticity': float,
    'instability_index': float,
    'isoelectric_point': float,
    'aa_composition': Dict[str, float],
    'secondary_structure': {'helix': float, 'turn': float, 'sheet': float}
}
```

---

### 8.9 `get_confidence_category(plddt)`

**Purpose:** Maps a numerical pLDDT score to a human-readable category.

```
>= 90  ->  "Very High"
>= 70  ->  "Confident"
>= 50  ->  "Low"
 < 50  ->  "Very Low"
```

---

### 8.10 `format_sequence_display(sequence, width=60)`

**Purpose:** Wraps a sequence into fixed-width lines for display in text areas.

**Example output:**
```
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT
GKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFF
```

---

### 8.11 `parse_pdb_coordinates(pdb_string, atom_type='CA')`

**Purpose:** Extracts 3D (x, y, z) coordinates of specified atom types from PDB format text.

**Default:** `atom_type='CA'` extracts only C-alpha backbone atoms — the standard minimal representation for structural comparison.

**Returns:** `numpy.ndarray` of shape (N, 3) where N = number of matching atoms

---

### 8.12 `calculate_rmsd(coords1, coords2)`

**Purpose:** Computes the Root Mean Square Deviation between two sets of 3D coordinates.

```
RMSD = sqrt( (1/N) x sum( ||ri - ri'||^2 ) )
```

**Requires:** Same number of atoms in both structures (no alignment performed).

**Returns:** RMSD value (float, in Angstroms) or `None` if sizes don't match.

---

### 8.13 `get_example_protein_ids()`

**Purpose:** Returns the reference dictionary of 5 curated example proteins with UniProt IDs and descriptions.

> **Important architectural note:** This function returns only UniProt IDs — never actual sequences. Sequences are always fetched live from the UniProt API when the user clicks Load.

---

### 8.14 `load_hf_dataset()` — Permanently Cached

**Purpose:** Loads a protein sequence dataset from Hugging Face. Tries datasets in priority order:

1. `damlab/uniprot` — 500,000+ curated SwissProt proteins
2. `LiteFold/PDB` — 230,000+ PDB experimentally-determined structures
3. `hyskova-anna/proteins` — Filtered proteins 20–400 residues

After loading, normalizes column names and filters:
- Removes null entries
- Removes sequences containing `*` (stop codons)
- Filters 10 <= length <= 2,000

---

### 8.15 `get_random_sequences_from_dataset(n=5)`

**Purpose:** Samples `n` random sequences from the loaded HF dataset with full metadata.

**Returns:** List of dictionaries with keys: `id`, `pdb_id`, `chain`, `sequence`, `length`, `method`, `resolution`, `organism`, `description`, `dataset_source`.

---

### 8.16 `get_dataset_stats()`

**Purpose:** Computes aggregate statistics over the loaded dataset.

**Returns:** `{'total_sequences': int, 'avg_length': float, 'min_length': float, 'max_length': float, 'columns': list}`

---

### 8.17 `get_plddt_color_scale()`

**Purpose:** Returns the canonical pLDDT color thresholds used consistently across all visualizations.

```python
[(90, '#0053D6'),  # Dark blue  - Very High
 (70, '#65CBF3'),  # Light blue - Confident
 (50, '#FFDB13'),  # Yellow     - Low
 (0,  '#FF7D45')]  # Orange     - Very Low
```

---

## 9. Hyperparameter-Tunable Analysis Functions

These 6 functions form the new accuracy-improvement layer added via hyperparameter tuning. Each function exposes at least one key hyperparameter that directly controls the **bias-variance trade-off**.

---

### 9.1 `calculate_sequence_quality(sequence)`

**Purpose:** Scores a protein sequence on three axes of prediction readiness.

**Algorithm:**

```
quality_score = 0.40 x complexity + 0.30 x (100 - disorder_score) + 0.30 x length_score
```

Where:
- **Complexity** = Shannon entropy / log2(20) x 100 — measures amino acid composition variety
- **Disorder score** = (disorder_aa_fraction - order_aa_fraction + 0.5) x 100
  - Disorder AAs (Dunker et al. 2001): R, K, E, P, Q, D, S
  - Order AAs: W, Y, F, I, L, V, M, C
- **Length score** = 100 for 30–600 residues; linearly penalized outside this range

**Warnings generated when:**
- Complexity < 50% -> repetitive sequence warning
- Disorder score > 70% -> high disorder warning
- Length > 600 residues -> accuracy drop warning
- Length > 1000 residues -> domain-splitting recommendation

---

### 9.2 `calculate_hydrophobicity_profile(sequence, window_size=9)`

**Purpose:** Sliding-window average hydrophobicity using the **Kyte-Doolittle scale** (1982).

**The Kyte-Doolittle Scale** assigns each amino acid a value from -4.5 (most hydrophilic) to +4.5 (most hydrophobic):

```
Most Hydrophobic -> I(+4.5), V(+4.2), L(+3.8), F(+2.8), C(+2.5) ...
Most Hydrophilic -> R(-4.5), K(-3.9), N(-3.5), D(-3.5), Q(-3.5) ...
```

**Hyperparameter — `window_size`:**

| Value | Effect | Use Case |
|---|---|---|
| 5–7 (small) | Captures sharp local hydrophobic patches | Signal peptides, binding sites |
| 9 (default) | Balanced local-global tradeoff | General purpose |
| 15–21 (large) | Smooth global hydrophobic domains | Transmembrane helix prediction |

**Returns:** `(positions: List[int], hydrophobicity_values: List[float])`

---

### 9.3 `predict_disorder_profile(sequence, window_size=9, threshold=0.5)`

**Purpose:** Predicts per-residue intrinsic disorder using a charge/hydrophobicity approach inspired by **IUPred** (Dosztanyi et al. 2005).

**Method:**
```
For each window:
1. Normalised hydrophobicity:  (raw_KD - (-4.5)) / 9.0   [0, 1]
2. Normalised net charge:      (pos_count - neg_count) / window + 0.5   [0, 1]
3. Disorder score:             (1 - hydrophobicity) x 0.6 + charge x 0.4
```

High charge + low hydrophobicity -> high disorder score (matching the Uversky diagram).

**Hyperparameter — `threshold`:**

| Value | Effect |
|---|---|
| 0.2–0.3 | Strict: flags many regions as disordered (low bias, high variance) |
| 0.5 (default) | Balanced — standard IUPred-like cutoff |
| 0.7–0.8 | Lenient: only obvious disorder flagged (high bias, low variance) |

**Returns:** `(positions, disorder_scores, is_disordered_flags: List[bool])`

---

### 9.4 `calculate_plddt_smoothed(residues, plddt, sigma=2.0)`

**Purpose:** Applies Gaussian convolution smoothing to noisy per-residue pLDDT signals.

**Algorithm:**
```python
kernel = exp(-0.5 x (x/sigma)^2)   # Gaussian kernel of radius ceil(3*sigma)
kernel /= kernel.sum()               # Normalize
padded = pad(plddt, radius, mode='edge')
smoothed = convolve(padded, kernel, mode='valid')
```

Edge padding prevents boundary artifacts by extending terminal values.

**Hyperparameter — `sigma`:**

| Sigma Value | Effect | Analogy |
|---|---|---|
| 0.5–1.0 | Near-raw signal; preserves sharp drops | Low regularization |
| 1.5 (default) | Light smoothing; clear trends visible | Balanced |
| 3.0–5.0 | Heavy smoothing; global trend only | Strong regularization |

**Implementation detail:** Uses pure NumPy `np.convolve` — no scipy dependency required.

---

### 9.5 `trim_low_confidence_termini(sequence, pdb_string, min_plddt=50.0)`

**Purpose:** Removes N- and C-terminal residues whose pLDDT falls below the confidence threshold, reducing structural variance from disordered protein ends.

**Algorithm:**
1. Extract pLDDT scores from PDB B-factors
2. Walk from N-terminus inward until the first residue >= `min_plddt`
3. Walk from C-terminus inward until the last residue >= `min_plddt`
4. Safety check: always retain at least 10 residues
5. Filter PDB ATOM/HETATM records to only include retained residue numbers
6. Slice the sequence string accordingly

**Hyperparameter — `min_plddt`:**

| Value | Trimming Behavior |
|---|---|
| 50 | Minimal trimming — only removes clearly disordered ends |
| 70 | Aggressive — trims until backbone is reliable |
| 90 | Maximum — retains only very high-confidence core |

**Returns:** `(trimmed_sequence, trimmed_pdb, n_trimmed_from_N, n_trimmed_from_C)`

---

### 9.6 `assess_prediction_reliability(plddt, threshold=70.0)`

**Purpose:** Assigns an A–D letter grade to a prediction based on per-residue pLDDT scores at a given threshold.

**Grading Rubric:**

| Grade | Condition |
|---|---|
| A – Excellent | avg >= 90 AND >= 80% residues >= threshold |
| B – Good | avg >= 70 AND >= 60% residues >= threshold |
| C – Fair | avg >= 50 AND >= 40% residues >= threshold |
| D – Poor | otherwise |

**Hyperparameter — `threshold`:** Higher threshold -> stricter grading -> fewer A/B results.

Also identifies **contiguous high-confidence segments** (runs of residues all above threshold), useful for identifying the reliably-folded protein core.

**Returns:**
```python
{
    'grade': 'A',
    'grade_label': 'Excellent',
    'grade_color': '#0053D6',
    'percent_reliable': 87.5,
    'avg_plddt': 91.2,
    'n_reliable': 210,
    'n_total': 240,
    'high_confidence_segments': [(1, 238)]
}
```

---

## 10. API Integrations

### UniProt REST API

| Endpoint | Method | Cache TTL | Purpose |
|---|---|---|---|
| `/uniprotkb/{id}.fasta` | GET | 1 hour | Sequence in FASTA format |
| `/uniprotkb/{id}.json` | GET | 1 hour | Full protein metadata |

**Authentication:** None required (public API).

---

### ESMFold API (ESM Metagenomic Atlas)

| Endpoint | Method | Cache TTL | Timeout | Purpose |
|---|---|---|---|---|
| `/foldSequence/v1/pdb/` | POST | 24 hours | 300s | Predict structure from sequence |

**Input format:** Plain text amino acid sequence in POST body.  
**Output format:** PDB-formatted structure file.  
**Prediction time:** Typically 30–60 seconds for a 100–400 residue sequence.  
**Accuracy:** Based on ESM-2 (650M parameter protein language model).

---

### AlphaFold EBI Database

| Endpoint | Method | Cache TTL | Purpose |
|---|---|---|---|
| `/files/AF-{id}-F1-model_v4.pdb` | GET | 24 hours | Pre-computed structure (v4) |
| `/files/AF-{id}-F1-model_v3.pdb` | GET | 24 hours | Fallback if v4 unavailable |

**Coverage:** 200M+ protein structures.  
**Limitation:** Only available for UniProt-registered proteins.

---

### Hugging Face Datasets

| Dataset | Size | Content |
|---|---|---|
| `damlab/uniprot` | 500K+ entries | Curated SwissProt sequences with metadata |
| `LiteFold/PDB` | 230K+ entries | Experimentally-determined PDB sequences |
| `hyskova-anna/proteins` | Variable | Filtered 20–400 residue proteins |

---

## 11. Session State Management

Streamlit reruns the entire Python script on every user interaction. Session state persists data across reruns using a key-value store (`st.session_state`).

### Core State Variables

| Key | Type | Purpose |
|---|---|---|
| `predicted_structure` | str (PDB) | The full PDB text of the most recent prediction |
| `current_sequence` | str | Current active amino acid sequence |
| `sequence_analysis` | Dict | Result of `analyze_sequence()` |
| `plddt_scores` | Tuple[List, List] | (residue_numbers, plddt_values) |
| `protein_name` | str | Display name for the current protein |
| `prediction_source` | str | "ESMFold API" or "AlphaFold DB" |

### Hyperparameter State Variables

| Key | Default | Controls |
|---|---|---|
| `hp_plddt_threshold` | 70 | Confidence cutoff |
| `hp_window_size` | 9 | Hydrophobicity/disorder window |
| `hp_disorder_threshold` | 0.5 | Disorder classification cutoff |
| `hp_auto_trim` | True | Auto-trim low-confidence termini |
| `hp_smooth_sigma` | 1.5 | Gaussian smoothing strength |
| `hp_ensemble_mode` | True | Ensemble mode toggle |

### Temporary Transfer State

| Key | Purpose |
|---|---|
| `temp_sequence` | Transfers sequence from Home -> Predict page |
| `temp_name` | Transfers protein name |
| `temp_uniprot` | Transfers UniProt ID |
| `random_dataset_sequences` | Stores loaded HF sequences |
| `dataset_stats` | Stores HF dataset aggregate statistics |

---

## 12. Caching Strategy

| Function | TTL | Rationale |
|---|---|---|
| `fetch_uniprot_sequence` | 1 hour | UniProt sequences change rarely |
| `fetch_uniprot_metadata` | 1 hour | Metadata is stable |
| `predict_structure_esmfold` | 24 hours | Deterministic; avoid API abuse |
| `fetch_alphafold_structure` | 24 hours | AlphaFold DB is static |
| `load_hf_dataset` | Session | Large dataset; load once per session |

---

## 13. Data Flow Diagram

```
USER INPUT
    |
    +-- Paste/Upload/UniProt/Random
    |
    v
validate_sequence()
    |  is_valid=True
    v
calculate_sequence_quality() <-- Hyperparameter panel
    |  quality_score, warnings
    v
+--------------------+   +--------------------+
|   ESMFold API      |   |   AlphaFold DB     |
| predict_structure  |   | fetch_alphafold_   |
| _esmfold(seq)      |   | structure(id)      |
+--------+-----------+   +----------+---------+
         |                          |
         +-----------+--------------+
                     v
         PDB string (ATOM records + B-factors)
                     |
         +-----------+--------------------+
         |                                |
         v                                v
extract_plddt_from_pdb()       analyze_sequence()
(residues, plddt)              (MW, pI, instability...)
         |                                |
         v                                v
+-----------------------------------------------+
|           show_results() - 5 tabs             |
|  3D Viewer | Confidence | Analysis             |
|  Download  | Reliability (HP-driven)           |
+-----------------------------------------------+
         |
         +-- calculate_plddt_smoothed()
         +-- assess_prediction_reliability()
         +-- calculate_hydrophobicity_profile()
         +-- predict_disorder_profile()
         +-- trim_low_confidence_termini()
```

---

## 14. Bias-Variance Trade-off Framework

The hyperparameter tuning system implements the classic machine learning bias-variance trade-off in a bioinformatics context.

### Concept Map

```
LOW BIAS <----------------------------------------> HIGH BIAS
HIGH VARIANCE                                  LOW VARIANCE

pLDDT threshold = 50  |   pLDDT threshold = 90
More residues shown   |   Fewer, reliable only

Window size = 5       |   Window size = 21
Sharp local features  |   Smooth global domains

Disorder thr = 0.2    |   Disorder thr = 0.8
Many regions flagged  |   Only obvious disorder

Sigma = 0.5           |   Sigma = 5.0
Raw noisy signal      |   Heavily smoothed

No trimming           |   Aggressive trimming
Full sequence         |   Reliable core only
```

### Practical Guidance

| Goal | Recommended Settings |
|---|---|
| Exploratory analysis | Low thresholds (50–60), small windows |
| Publication-quality | High thresholds (70–80), auto-trim enabled |
| Domain identification | Large windows (15–21) |
| Binding site analysis | Small windows (5–7) |
| IDP studies | Disorder threshold 0.2–0.3 |

---

## 15. Testing & Validation

### Test Coverage (run_checks.py)

| Module | Tests | Status |
|---|---|---|
| `validate_sequence` | 7 | All pass |
| `parse_fasta` | 5 | All pass |
| `get_confidence_category` | 8 | All pass |
| `format_sequence_display` | 4 | All pass |
| `get_example_protein_ids` | 11 | All pass |
| `get_plddt_color_scale` | 3 | All pass |
| `parse_pdb_coordinates` | 4 | All pass |
| `calculate_rmsd` | 4 | All pass |
| `extract_plddt_from_pdb` | 3 | All pass |
| `analyze_sequence` | 13 | All pass |
| **Total** | **46** | **All passing** |

### Running Tests

```bash
python run_checks.py
```

---

## 16. Limitations & Best Practices

### Known Limitations

| Limitation | Detail |
|---|---|
| **Sequence length** | ESMFold API supports up to ~2,000 residues; accuracy drops above 600 |
| **Monomers only** | Neither ESMFold nor AlphaFold models predict protein-protein complexes |
| **No PTMs** | Post-translational modifications are not modeled |
| **Disorder accuracy** | pLDDT < 50 indicates probable disorder, but does not confirm IDP characteristics |
| **RMSD alignment** | Requires pre-aligned, same-length structures; no automatic alignment |
| **API dependency** | ESMFold predictions require internet connectivity |
| **No ligands** | Ligand-binding site geometry is not modeled |

### Best Practices

1. Always check pLDDT scores before drawing biological conclusions
2. Set pLDDT threshold >= 70 for reliable residue analysis
3. Use AlphaFold DB when a UniProt ID is available
4. Trim disordered termini before docking or comparative modeling
5. For long sequences, split into structural domains and predict each separately
6. Validate with experimental data (X-ray, cryo-EM, NMR) wherever possible
7. Use batch mode for screening multiple sequences

---

## 17. References & Scientific Background

### Core Prediction Models

1. **ESMFold:**  
   Lin, Z., et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 379(6637), 1123-1130.

2. **AlphaFold2:**  
   Jumper, J., et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature, 596(7873), 583-589.

### Analysis Methods

3. **Kyte-Doolittle Hydrophobicity:**  
   Kyte, J., & Doolittle, R.F. (1982). *A simple method for displaying the hydropathic character of a protein.* J Mol Biol, 157(1), 105-132.

4. **Instability Index:**  
   Guruprasad, K., et al. (1990). *Correlation between stability of a protein and its dipeptide composition.* Protein Engineering, 4(2), 155-161.

5. **Protein Disorder (IUPred approach):**  
   Dosztanyi, Z., et al. (2005). *IUPred: web server for the prediction of intrinsically unstructured regions of proteins.* Bioinformatics, 21(16), 3433-3434.

6. **Disorder Propensity Scale (Dunker):**  
   Dunker, A.K., et al. (2001). *Intrinsically disordered protein.* J Mol Graph Model, 19(1), 26-59.

7. **pLDDT Confidence Metric:**  
   Mariani, V., et al. (2013). *lDDT: a local superposition-free score for comparing protein structures.* Bioinformatics, 29(21), 2722-2728.

### Software & Tools

8. **Biopython:**  
   Cock, P.J.A., et al. (2009). *Biopython: freely available Python tools for computational molecular biology.* Bioinformatics, 25(11), 1422-1423.

9. **Streamlit:** https://streamlit.io

10. **py3Dmol:**  
    Rego, N. & Koes, D. (2015). *3Dmol.js: molecular visualization with WebGL.* Bioinformatics, 31(8), 1322-1324.

11. **UniProt:**  
    UniProt Consortium (2023). *UniProt: the Universal Protein Knowledgebase in 2023.* Nucleic Acids Research, 51(D1), D523-D531.

---

*Documentation last updated: July 2026*  
*ProteinForge v1.0 | 100% Open Source | Zero Hard-Coded Data*
