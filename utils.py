"""
Dynamic Utility Functions for ProteinForge
All data fetched from APIs - NO hard-coded sequences or structures
"""
import requests
import time
import streamlit as st
from io import StringIO
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Optional, Tuple, Dict, List
import re


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate amino acid sequence.
    
    Args:
        sequence: Amino acid sequence string
        
    Returns:
        Tuple of (is_valid, message)
    """
    sequence = sequence.upper().strip()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    if not sequence:
        return False, "Sequence is empty"
    
    # Remove whitespace and newlines
    sequence_clean = sequence.replace(" ", "").replace("\n", "").replace("\r", "")
    sequence_set = set(sequence_clean)
    
    invalid_chars = sequence_set - valid_aa
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(sorted(invalid_chars))}"
    
    if len(sequence_clean) < 10:
        return False, "Sequence too short (minimum 10 residues)"
    
    if len(sequence_clean) > 2000:
        return False, "Sequence too long (maximum 2000 residues for ESMFold API)"
    
    return True, sequence_clean


def parse_fasta(fasta_content: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA file content and return list of (name, sequence) tuples.
    
    Args:
        fasta_content: FASTA formatted string
        
    Returns:
        List of tuples containing (sequence_name, sequence)
    """
    sequences = []
    try:
        fasta_io = StringIO(fasta_content)
        for record in SeqIO.parse(fasta_io, "fasta"):
            seq_str = str(record.seq).upper()
            # Clean sequence
            seq_clean = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq_str)
            if len(seq_clean) >= 10:
                sequences.append((record.id, seq_clean))
    except Exception as e:
        st.error(f"Error parsing FASTA: {str(e)}")
        return []
    return sequences


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_sequence(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch protein sequence from UniProt API (cached for 1 hour).
    
    Args:
        uniprot_id: UniProt accession number
        
    Returns:
        Tuple of (sequence, protein_name) or (None, None) if failed
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            fasta_content = response.text
            sequences = parse_fasta(fasta_content)
            if sequences:
                return sequences[0][1], sequences[0][0]
        return None, None
    except Exception as e:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_metadata(uniprot_id: str) -> Optional[Dict]:
    """
    Fetch protein metadata from UniProt API.
    
    Args:
        uniprot_id: UniProt accession number
        
    Returns:
        Dictionary with metadata or None if failed
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract protein name
            protein_name = "Unknown"
            if 'proteinDescription' in data:
                if 'recommendedName' in data['proteinDescription']:
                    if 'fullName' in data['proteinDescription']['recommendedName']:
                        protein_name = data['proteinDescription']['recommendedName']['fullName'].get('value', 'Unknown')
            
            # Extract organism
            organism = data.get('organism', {}).get('scientificName', 'Unknown')
            
            # Extract gene name
            gene = 'Unknown'
            if 'genes' in data and len(data['genes']) > 0:
                if 'geneName' in data['genes'][0]:
                    gene = data['genes'][0]['geneName'].get('value', 'Unknown')
            
            # Extract length
            length = data.get('sequence', {}).get('length', 0)
            
            return {
                'name': protein_name,
                'organism': organism,
                'gene': gene,
                'length': length
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def predict_structure_esmfold(sequence: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Predict protein structure using ESMFold API (cached for 24 hours).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Tuple of (pdb_string, error_message)
    """
    try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {'Content-Type': 'text/plain'}
        response = requests.post(url, data=sequence, headers=headers, timeout=300)
        
        if response.status_code == 200:
            pdb_string = response.text
            if len(pdb_string) > 100 and 'ATOM' in pdb_string:
                return pdb_string, None
            else:
                return None, "Invalid PDB response from ESMFold"
        elif response.status_code == 400:
            return None, "Invalid sequence format"
        elif response.status_code == 413:
            return None, "Sequence too long for ESMFold API"
        else:
            return None, f"ESMFold API error: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Request timed out (>5 minutes). Try a shorter sequence."
    except requests.exceptions.ConnectionError:
        return None, "Connection failed. Check internet connection."
    except Exception as e:
        return None, f"Error: {str(e)}"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_alphafold_structure(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch pre-computed AlphaFold structure from AlphaFold DB (cached for 24 hours).
    
    Args:
        uniprot_id: UniProt accession number
        
    Returns:
        Tuple of (pdb_string, error_message)
    """
    try:
        # Try v4 first
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            pdb_string = response.text
            if len(pdb_string) > 100 and 'ATOM' in pdb_string:
                return pdb_string, None
        
        # Try v3 as fallback
        url_v3 = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb"
        response = requests.get(url_v3, timeout=30)
        
        if response.status_code == 200:
            pdb_string = response.text
            if len(pdb_string) > 100 and 'ATOM' in pdb_string:
                return pdb_string, None
        
        return None, f"No AlphaFold structure available for {uniprot_id}"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"


def extract_plddt_from_pdb(pdb_string: str) -> Tuple[List[int], List[float]]:
    """
    Extract pLDDT scores from PDB file (stored in B-factor column).
    
    Args:
        pdb_string: PDB file content as string
        
    Returns:
        Tuple of (residue_numbers, plddt_scores)
    """
    plddt_scores = []
    residue_numbers = []
    
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # Only CA atoms
            try:
                residue_num = int(line[22:26].strip())
                plddt = float(line[60:66].strip())
                residue_numbers.append(residue_num)
                plddt_scores.append(plddt)
            except (ValueError, IndexError):
                continue
    
    return residue_numbers, plddt_scores


def analyze_sequence(sequence: str) -> Optional[Dict]:
    """
    Analyze sequence composition and properties.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        analyzer = ProteinAnalysis(sequence)
        
        aa_composition = analyzer.amino_acids_percent
        molecular_weight = analyzer.molecular_weight()
        aromaticity = analyzer.aromaticity()
        instability = analyzer.instability_index()
        isoelectric_point = analyzer.isoelectric_point()
        
        # Secondary structure fraction (rough estimate)
        try:
            helix, turn, sheet = analyzer.secondary_structure_fraction()
        except:
            helix, turn, sheet = 0.0, 0.0, 0.0
        
        return {
            'length': len(sequence),
            'molecular_weight': molecular_weight,
            'aromaticity': aromaticity,
            'instability_index': instability,
            'isoelectric_point': isoelectric_point,
            'aa_composition': aa_composition,
            'secondary_structure': {
                'helix': helix,
                'turn': turn,
                'sheet': sheet
            }
        }
    except Exception as e:
        return None


def get_confidence_category(plddt: float) -> str:
    """
    Get confidence category from pLDDT score.
    
    Args:
        plddt: pLDDT score (0-100)
        
    Returns:
        Confidence category string
    """
    if plddt >= 90:
        return "Very High"
    elif plddt >= 70:
        return "Confident"
    elif plddt >= 50:
        return "Low"
    else:
        return "Very Low"


def format_sequence_display(sequence: str, width: int = 60) -> str:
    """
    Format sequence for display with line breaks.
    
    Args:
        sequence: Amino acid sequence
        width: Characters per line
        
    Returns:
        Formatted sequence string
    """
    return '\n'.join([sequence[i:i+width] for i in range(0, len(sequence), width)])


def parse_pdb_coordinates(pdb_string: str, atom_type: str = 'CA') -> np.ndarray:
    """
    Extract coordinates from PDB file.
    
    Args:
        pdb_string: PDB file content
        atom_type: Atom type to extract (default: CA for C-alpha)
        
    Returns:
        Numpy array of coordinates
    """
    coords = []
    
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == atom_type:
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
            except (ValueError, IndexError):
                continue
    
    return np.array(coords)


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    """
    Calculate RMSD between two sets of coordinates.
    
    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates
        
    Returns:
        RMSD value or None if calculation fails
    """
    if len(coords1) != len(coords2) or len(coords1) == 0:
        return None
    
    try:
        diff = coords1 - coords2
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    except:
        return None


def get_example_protein_ids() -> Dict[str, Dict[str, str]]:
    """
    Return list of UniProt IDs for dynamic loading of example proteins.
    NO hard-coded sequences - all fetched via API when user clicks.
    
    Returns:
        Dictionary of example proteins with UniProt IDs and descriptions
    """
    return {
        "Insulin (Human)": {
            "uniprot": "P01308",
            "description": "Hormone regulating glucose metabolism"
        },
        "Hemoglobin Alpha": {
            "uniprot": "P69905",
            "description": "Oxygen transport protein"
        },
        "Green Fluorescent Protein": {
            "uniprot": "P42212",
            "description": "Fluorescent protein from jellyfish"
        },
        "Lysozyme": {
            "uniprot": "P00698",
            "description": "Antimicrobial enzyme"
        },
        "Myoglobin": {
            "uniprot": "P02144",
            "description": "Oxygen storage in muscles"
        }
    }


@st.cache_data(show_spinner=False)
def load_hf_dataset() -> Optional[pd.DataFrame]:
    """
    Load protein dataset from Hugging Face.
    Uses high-quality protein sequence datasets.
    
    Returns:
        DataFrame or None if loading fails
    """
    try:
        from datasets import load_dataset
        
        # Try multiple high-quality protein datasets in order of preference
        dataset_options = [
            ("damlab/uniprot", "train", "Best: 500K+ curated UniProt/SwissProt proteins"),
            ("LiteFold/PDB", "train", "230K+ PDB experimentally-determined structures"),
            ("hyskova-anna/proteins", "train", "Filtered proteins (20-400 residues)"),
        ]
        
        df = None
        loaded_dataset_name = None
        
        for dataset_name, split, description in dataset_options:
            try:
                st.info(f"📦 Loading: {description}")
                ds = load_dataset(dataset_name, split=split)
                df = ds.to_pandas()
                loaded_dataset_name = dataset_name
                
                # Check if it has sequence data
                seq_col = None
                for col in ['sequence', 'seq', 'Sequence', 'SEQ', 'primary', 'aa_sequence']:
                    if col in df.columns:
                        seq_col = col
                        break
                
                if seq_col:
                    # Rename to standard 'seq' for consistency
                    if seq_col != 'seq':
                        df = df.rename(columns={seq_col: 'seq'})
                    
                    # Filter valid sequences
                    df = df[df['seq'].notna()].copy()
                    df = df[~df['seq'].astype(str).str.contains(r'\*', na=False)].copy()
                    df = df[df['seq'].astype(str).str.len() >= 10].copy()  # Minimum length
                    df = df[df['seq'].astype(str).str.len() <= 2000].copy()  # Maximum length for ESMFold
                    
                    st.success(f"✅ Loaded {len(df):,} sequences from {dataset_name}")
                    break
                else:
                    st.warning(f"No sequence column in {dataset_name}")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load {dataset_name}: {str(e)}")
                continue
        
        if df is None or len(df) == 0:
            st.error("❌ Could not load any protein dataset from Hugging Face")
            return None
        
        # Add dataset source to the dataframe
        df['dataset_source'] = loaded_dataset_name
        
        return df
        
    except Exception as e:
        st.error(f"Error loading HF dataset: {str(e)}")
        return None


def get_random_sequences_from_dataset(n: int = 5) -> List[Dict[str, str]]:
    """
    Get random protein sequences from the Hugging Face dataset.
    
    Args:
        n: Number of sequences to return
        
    Returns:
        List of dictionaries with sequence info
    """
    df = load_hf_dataset()
    if df is None or len(df) == 0:
        return []
    
    # Ensure 'seq' column exists
    if 'seq' not in df.columns:
        st.error("'seq' column not found after loading.")
        return []
    
    # Filter out invalid sequences
    df = df[df['seq'].notna()].copy()
    df = df[df['seq'].astype(str).str.len() >= 10].copy()
    
    if len(df) == 0:
        st.warning("No valid sequences found in dataset")
        return []
    
    # Sample random sequences
    sample_df = df.sample(n=min(n, len(df)))
    results = []
    
    for idx, row in sample_df.iterrows():
        sequence = str(row['seq'])
        
        # Skip if sequence is empty or invalid
        if not sequence or sequence == 'nan' or len(sequence) < 10:
            continue
        
        # Get ID from various possible columns (damlab/uniprot uses 'name')
        seq_id = None
        for id_col in ['name', 'id', 'pdb_id', 'entry', 'accession']:
            if id_col in row and pd.notna(row[id_col]):
                seq_id = str(row[id_col])
                break
        
        if not seq_id:
            seq_id = f'seq_{idx}'
        
        # Get organism if available (damlab/uniprot has 'organism')
        organism = None
        for org_col in ['organism', 'species', 'organism_name']:
            if org_col in row and pd.notna(row[org_col]):
                organism = str(row[org_col])
                break
        
        # Get other metadata
        chain = str(row.get('chain_code', row.get('chain', '')))
        length = len(sequence)
        
        dataset_source = str(row.get('dataset_source', 'HuggingFace'))
        
        # Create display ID
        if chain and chain != 'nan' and chain != '':
            full_id = f"{seq_id}_{chain}"
        else:
            full_id = seq_id
        
        # Get method and resolution if available (PDB datasets)
        method = 'Unknown'
        for method_col in ['Exptl.', 'method', 'resolution_method', 'experiment_type']:
            if method_col in row and pd.notna(row[method_col]):
                method = str(row[method_col])
                break
        
        resolution = 'N/A'
        if 'resolution' in row and pd.notna(row['resolution']):
            resolution = str(row['resolution'])
        
        # Build description
        description_parts = []
        if organism and organism != 'nan':
            description_parts.append(organism)
        description_parts.append(f"{length} aa")
        
        results.append({
            'id': full_id,
            'pdb_id': seq_id,
            'chain': chain if chain != 'nan' else '',
            'sequence': sequence,
            'length': length,
            'method': method if method != 'nan' else 'Unknown',
            'resolution': resolution if resolution != 'nan' else 'N/A',
            'organism': organism if organism and organism != 'nan' else 'Unknown',
            'description': ' | '.join(description_parts),
            'dataset_source': dataset_source
        })
    
    return results


def search_dataset_by_length(min_length: int = 20, max_length: int = 100, limit: int = 10) -> List[Dict[str, str]]:
    """
    Search Hugging Face dataset by sequence length.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length  
        limit: Maximum number of results
        
    Returns:
        List of sequence dictionaries
    """
    df = load_hf_dataset()
    if df is not None:
        # Get length column name
        len_col = 'len' if 'len' in df.columns else 'length'
        
        if len_col in df.columns:
            filtered = df[(df[len_col] >= min_length) & (df[len_col] <= max_length)]
            filtered = filtered.head(limit)
            
            results = []
            for _, row in filtered.iterrows():
                seq_id = row.get('pdb_id', row.get('id', 'Unknown'))
                chain = row.get('chain_code', row.get('chain', ''))
                sequence = row.get('seq', row.get('sequence', ''))
                
                results.append({
                    'id': f"{seq_id}_{chain}" if chain else seq_id,
                    'sequence': sequence,
                    'length': int(row[len_col])
                })
            return results
    return []


def get_dataset_stats() -> Optional[Dict[str, any]]:
    """
    Get statistics about the loaded dataset.
    
    Returns:
        Dictionary with dataset statistics or None
    """
    df = load_hf_dataset()
    if df is not None and len(df) > 0:
        # Filter out invalid sequences
        df = df[df['seq'].notna()].copy()
        df = df[df['seq'].str.len() > 0].copy()
        
        len_col = 'len' if 'len' in df.columns else 'length'
        seq_col = 'seq' if 'seq' in df.columns else 'sequence'
        
        # Calculate lengths from actual sequences if len column missing or has NaN
        if len_col not in df.columns or df[len_col].isna().any():
            df['calculated_len'] = df[seq_col].str.len()
            len_col = 'calculated_len'
        
        stats = {
            'total_sequences': len(df),
            'avg_length': float(df[len_col].mean()) if len_col in df.columns else 0,
            'min_length': float(df[len_col].min()) if len_col in df.columns else 0,
            'max_length': float(df[len_col].max()) if len_col in df.columns else 0,
            'columns': list(df.columns)
        }
        return stats
    return None


def get_plddt_color_scale() -> List[Tuple[float, str]]:
    """
    Get pLDDT color scale for visualization.
    
    Returns:
        List of (threshold, color) tuples
    """
    return [
        (90, '#0053D6'),  # Dark blue - very high
        (70, '#65CBF3'),  # Light blue - confident
        (50, '#FFDB13'),  # Yellow - low
        (0, '#FF7D45')    # Orange - very low
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER-TUNABLE ANALYSIS FUNCTIONS
# Each function exposes a key bias-variance trade-off knob as a hyperparameter.
# ═══════════════════════════════════════════════════════════════════════════════

# Kyte-Doolittle hydrophobicity scale (standard literature values)
KYTE_DOOLITTLE: Dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}


def calculate_sequence_quality(sequence: str) -> Dict:
    """
    Estimate prediction readiness by scoring three factors:
    1. Shannon entropy  (sequence complexity).
    2. Disorder propensity  (Dunker charge/hydrophobicity balance).
    3. Length penalty  (ESMFold accuracy drops beyond 600 residues).

    Returns:
        Dict with quality_score (0-100), complexity, disorder_score,
        length_score, and warnings (list of str).
    """
    n = len(sequence)
    if n == 0:
        return {
            'quality_score': 0, 'complexity': 0.0,
            'disorder_score': 0.0, 'length_score': 0.0, 'warnings': [],
        }

    # -- 1. Shannon entropy (complexity) --------------------------------------
    aa_counts: Dict[str, int] = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    entropy = -sum(
        (c / n) * float(np.log2(c / n))
        for c in aa_counts.values() if c > 0
    )
    max_entropy = float(np.log2(20))        # log2(20) ~ 4.32 for 20 amino acids
    complexity  = min((entropy / max_entropy) * 100.0, 100.0)

    # -- 2. Disorder propensity (Dunker et al. 2001) --------------------------
    disorder_aa = set('RKEPQDS')
    order_aa    = set('WYFILVMC')
    d_frac = sum(1 for aa in sequence if aa in disorder_aa) / n
    o_frac = sum(1 for aa in sequence if aa in order_aa)    / n
    # Map net disorder fraction to 0-100 (50 = neutral)
    disorder_score = min(max((d_frac - o_frac + 0.5) * 100.0, 0.0), 100.0)

    # -- 3. Length penalty ----------------------------------------------------
    if n > 600:
        length_score = max(0.0, 100.0 - (n - 600) * 0.05)
    elif n < 30:
        length_score = float(n) * 3.0
    else:
        length_score = 100.0

    # -- Weighted overall quality score --------------------------------------
    quality_score = int(
        0.40 * complexity
        + 0.30 * (100.0 - disorder_score)
        + 0.30 * length_score
    )
    quality_score = min(max(quality_score, 0), 100)

    # -- Actionable warnings -------------------------------------------------
    warnings: List[str] = []
    if complexity < 50.0:
        warnings.append(
            f"Low sequence complexity ({complexity:.1f}%) - "
            "repetitive/biased sequences often predict poorly with ESMFold."
        )
    if disorder_score > 70.0:
        warnings.append(
            "High disorder propensity - expect low pLDDT scores "
            "in disordered regions."
        )
    if n > 600:
        warnings.append(
            f"Long sequence ({n} aa) - ESMFold accuracy drops above 600 residues; "
            "consider predicting individual domains."
        )
    if n > 1000:
        warnings.append(
            "Very long sequence - domain splitting is strongly recommended."
        )

    return {
        'quality_score': quality_score,
        'complexity': complexity,
        'disorder_score': disorder_score,
        'length_score': length_score,
        'warnings': warnings,
    }


def calculate_hydrophobicity_profile(
    sequence: str,
    window_size: int = 9,
) -> Tuple[List[int], List[float]]:
    """
    Sliding-window hydrophobicity profile (Kyte-Doolittle scale).

    Hyperparameter - window_size (bias-variance trade-off):
      Small (5-7):   sharp local features; higher variance, lower bias.
      Large (15-21): smooth global trend; lower variance, higher bias.

    Args:
        sequence:    Amino acid sequence.
        window_size: Odd integer 5-21 (default 9).

    Returns:
        (positions, hydrophobicity_values) - both 1-indexed.
    """
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, min(window_size, len(sequence)))
    half = window_size // 2

    positions: List[int]   = []
    values:    List[float] = []
    for i in range(half, len(sequence) - half):
        win   = sequence[i - half: i + half + 1]
        avg_h = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in win) / window_size
        positions.append(i + 1)
        values.append(round(avg_h, 4))

    return positions, values


def predict_disorder_profile(
    sequence: str,
    window_size: int = 9,
    threshold: float = 0.5,
) -> Tuple[List[int], List[float], List[bool]]:
    """
    Per-residue disorder prediction via charge/hydrophobicity ratio
    (simplified IUPred-like heuristic).

    Hyperparameter - threshold (bias-variance trade-off):
      Low (0.2-0.3):  more windows flagged as disordered - low bias, high variance.
      High (0.7-0.8): only obvious disorder flagged - high bias, low variance.

    Args:
        sequence:    Amino acid sequence.
        window_size: Sliding window size (default 9).
        threshold:   Disorder score cutoff in [0, 1] (default 0.5).

    Returns:
        (positions, disorder_scores, is_disordered_flags)
    """
    pos_charged = set('RK')
    neg_charged = set('DE')

    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, min(window_size, len(sequence)))
    half = window_size // 2

    kd_min, kd_max = -4.5, 4.5
    positions: List[int]   = []
    scores:    List[float] = []
    flags:     List[bool]  = []

    for i in range(half, len(sequence) - half):
        win = sequence[i - half: i + half + 1]

        # Normalised hydrophobicity: 0 (hydrophilic) -> 1 (hydrophobic)
        raw_h   = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in win) / window_size
        hydro_n = (raw_h - kd_min) / (kd_max - kd_min)

        # Normalised net charge: 0 (neg) -> 1 (pos)
        net_ch   = (
            sum(1 for aa in win if aa in pos_charged)
            - sum(1 for aa in win if aa in neg_charged)
        ) / window_size
        charge_n = min(max(net_ch + 0.5, 0.0), 1.0)

        # High charge + low hydrophobicity -> high disorder
        d_score = min(max((1 - hydro_n) * 0.6 + charge_n * 0.4, 0.0), 1.0)

        positions.append(i + 1)
        scores.append(round(d_score, 4))
        flags.append(d_score >= threshold)

    return positions, scores, flags


def calculate_plddt_smoothed(
    residues: List[int],
    plddt: List[float],
    sigma: float = 2.0,
) -> List[float]:
    """
    Gaussian smoothing of per-residue pLDDT scores via numpy convolution.

    Hyperparameter - sigma (bias-variance trade-off):
      Small sigma (0.5-1.0): closely follows raw signal; low bias, high variance.
      Large sigma (3.0-5.0): heavy smoothing; high bias, low variance.

    Args:
        residues: Residue indices (kept for API symmetry with other functions).
        plddt:    Raw pLDDT scores.
        sigma:    Gaussian standard deviation (default 2.0).

    Returns:
        Smoothed scores clipped to [0, 100], same length as input.
    """
    if not plddt:
        return []

    arr    = np.array(plddt, dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x      = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded   = np.pad(arr, radius, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    smoothed = np.clip(smoothed, 0.0, 100.0)
    return smoothed.tolist()


def trim_low_confidence_termini(
    sequence: str,
    pdb_string: str,
    min_plddt: float = 50.0,
) -> Tuple[str, str, int, int]:
    """
    Remove N/C-terminal residues whose pLDDT falls below min_plddt.

    Disordered termini are the largest source of per-residue variance in
    ESMFold outputs. Trimming them trades a small coverage bias for a
    meaningful reduction in confidence variance.

    Hyperparameter - min_plddt:
      Low (50):  minimal trimming; low bias, higher variance.
      High (70): aggressive trimming; higher bias, lower variance.

    Args:
        sequence:   Original amino acid sequence.
        pdb_string: PDB file content.
        min_plddt:  pLDDT threshold for terminal retention (default 50.0).

    Returns:
        (trimmed_sequence, trimmed_pdb, n_trimmed_from_N, n_trimmed_from_C)
    """
    residues_list, plddt_vals = extract_plddt_from_pdb(pdb_string)

    if not plddt_vals or len(plddt_vals) < 5:
        return sequence, pdb_string, 0, 0

    # N-terminal: first residue meeting the threshold
    n_trim = len(plddt_vals)
    for i, s in enumerate(plddt_vals):
        if s >= min_plddt:
            n_trim = i
            break

    # C-terminal: last residue meeting the threshold
    c_trim = len(plddt_vals)
    for i in range(len(plddt_vals) - 1, -1, -1):
        if plddt_vals[i] >= min_plddt:
            c_trim = len(plddt_vals) - 1 - i
            break

    # Safety: always keep at least 10 residues
    if n_trim + c_trim >= len(sequence) - 10:
        return sequence, pdb_string, 0, 0

    end_idx     = len(sequence) - c_trim if c_trim > 0 else len(sequence)
    trimmed_seq = sequence[n_trim:end_idx]

    kept_start = n_trim
    kept_end   = len(residues_list) - c_trim if c_trim > 0 else len(residues_list)
    valid_res  = set(residues_list[kept_start:kept_end])

    out_lines: List[str] = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                res_num = int(line[22:26].strip())
                if res_num in valid_res:
                    out_lines.append(line)
            except (ValueError, IndexError):
                out_lines.append(line)
        else:
            out_lines.append(line)

    trimmed_pdb = '\n'.join(out_lines)
    return trimmed_seq, trimmed_pdb, n_trim, c_trim


def assess_prediction_reliability(
    plddt: List[float],
    threshold: float = 70.0,
) -> Dict:
    """
    Assign an A-D reliability grade to a structure prediction.

    Grading rubric:
      A (Excellent): avg >= 90 AND >= 80% of residues above threshold.
      B (Good):      avg >= 70 AND >= 60% above threshold.
      C (Fair):      avg >= 50 AND >= 40% above threshold.
      D (Poor):      otherwise.

    Hyperparameter - threshold:
      Higher values are stricter: fewer predictions earn A/B grades
      (precision-recall trade-off in quality assessment).

    Args:
        plddt:     Per-residue pLDDT scores.
        threshold: Minimum pLDDT for a residue to count as reliable.

    Returns:
        Dict with grade, grade_label, grade_color, percent_reliable,
        avg_plddt, n_reliable, n_total, high_confidence_segments.
    """
    if not plddt:
        return {
            'grade': 'N/A', 'grade_label': 'No Data', 'grade_color': '#888',
            'percent_reliable': 0.0, 'avg_plddt': 0.0,
            'n_reliable': 0, 'n_total': 0,
            'high_confidence_segments': [],
        }

    avg        = sum(plddt) / len(plddt)
    n_reliable = sum(1 for p in plddt if p >= threshold)
    pct        = n_reliable / len(plddt) * 100.0

    if avg >= 90 and pct >= 80:
        grade, label, color = 'A', 'Excellent', '#0053D6'
    elif avg >= 70 and pct >= 60:
        grade, label, color = 'B', 'Good',      '#10b981'
    elif avg >= 50 and pct >= 40:
        grade, label, color = 'C', 'Fair',      '#f59e0b'
    else:
        grade, label, color = 'D', 'Poor',      '#ef4444'

    # Contiguous high-confidence segments
    segments: List[Tuple[int, int]] = []
    in_seg, seg_start = False, 0
    for i, p in enumerate(plddt):
        if p >= threshold and not in_seg:
            in_seg, seg_start = True, i + 1   # 1-indexed
        elif p < threshold and in_seg:
            in_seg = False
            segments.append((seg_start, i))
    if in_seg:
        segments.append((seg_start, len(plddt)))

    return {
        'grade': grade,
        'grade_label': label,
        'grade_color': color,
        'percent_reliable': pct,
        'avg_plddt': avg,
        'n_reliable': n_reliable,
        'n_total': len(plddt),
        'high_confidence_segments': segments,
    }
