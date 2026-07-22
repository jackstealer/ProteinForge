"""
Detailed function-level analysis of ProteinForge utils.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from utils import (
    validate_sequence, parse_fasta, get_example_protein_ids,
    get_confidence_category, format_sequence_display,
    parse_pdb_coordinates, calculate_rmsd, get_plddt_color_scale,
    analyze_sequence, extract_plddt_from_pdb
)

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name} {detail}")
    else:
        print(f"  {FAIL} {name} {detail}")
        errors.append(name)

print("=" * 60)
print("ProteinForge Function Analysis")
print("=" * 60)

# ─────────────────────────────────────────────
print("\n[1] validate_sequence")
ok, msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
check("valid sequence returns True", ok)
check("valid sequence returns cleaned seq", msg == "ACDEFGHIKLMNPQRSTVWY")

ok2, msg2 = validate_sequence("ABC123!!")
check("invalid chars detected", not ok2)

ok3, msg3 = validate_sequence("MACYL")   # 5 chars < 10
check("too short rejected", not ok3)

ok4, msg4 = validate_sequence("A" * 2001)
check("too long rejected", not ok4)

ok5, msg5 = validate_sequence("")
check("empty string rejected", not ok5)

ok6, msg6 = validate_sequence("  MACY MACY\nMACY  ")   # spaces/newlines -> 12 chars
check("whitespace stripped properly", ok6, f"-> cleaned={msg6}")

# ─────────────────────────────────────────────
print("\n[2] parse_fasta")
fasta = ">P12345 Test protein\nMACDEFGHIK\nLMNPQRSTVW\n"
seqs = parse_fasta(fasta)
check("parses single fasta", len(seqs) == 1)
check("correct ID", seqs[0][0] == "P12345", f"-> got {seqs[0][0]}")
check("correct sequence", seqs[0][1] == "MACDEFGHIKLMNPQRSTVW", f"-> got {seqs[0][1]}")

multi_fasta = ">A\nACDEFGHIKL\n>B\nMNPQRSTVWY\n"
seqs2 = parse_fasta(multi_fasta)
check("parses multi-fasta", len(seqs2) == 2)

bad_fasta = "not a fasta at all"
seqs3 = parse_fasta(bad_fasta)
check("empty list on non-fasta", seqs3 == [])

# ─────────────────────────────────────────────
print("\n[3] get_confidence_category")
check(">=90 -> Very High", get_confidence_category(95) == "Very High")
check(">=90 boundary", get_confidence_category(90) == "Very High")
check("70-89 -> Confident", get_confidence_category(75) == "Confident")
check("70 boundary -> Confident", get_confidence_category(70) == "Confident")
check("50-69 -> Low", get_confidence_category(60) == "Low")
check("50 boundary -> Low", get_confidence_category(50) == "Low")
check("<50 -> Very Low", get_confidence_category(30) == "Very Low")
check("0 -> Very Low", get_confidence_category(0) == "Very Low")

# ─────────────────────────────────────────────
print("\n[4] format_sequence_display")
seq = "A" * 130
formatted = format_sequence_display(seq, width=60)
lines = formatted.split("\n")
check("130 chars -> 3 lines with width=60", len(lines) == 3, f"-> {len(lines)} lines")
check("first line is 60 chars", len(lines[0]) == 60)
check("last line is 10 chars", len(lines[2]) == 10)

short_seq = "MACDE"
formatted2 = format_sequence_display(short_seq)
check("short seq stays in one line", formatted2 == "MACDE")

# ─────────────────────────────────────────────
print("\n[5] get_example_protein_ids")
examples = get_example_protein_ids()
check("returns dict", isinstance(examples, dict))
check("has >= 5 examples", len(examples) >= 5, f"-> {len(examples)}")
for name, info in examples.items():
    check(f"  '{name}' has 'uniprot' key", "uniprot" in info)
    check(f"  '{name}' has 'description' key", "description" in info)

# ─────────────────────────────────────────────
print("\n[6] get_plddt_color_scale")
scale = get_plddt_color_scale()
check("returns list", isinstance(scale, list))
check("has 4 entries", len(scale) == 4)
thresholds = [t for t, _ in scale]
check("thresholds are 90,70,50,0", thresholds == [90, 70, 50, 0], f"-> {thresholds}")

# ─────────────────────────────────────────────
print("\n[7] parse_pdb_coordinates")
dummy_pdb = (
    "ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 85.00           C\n"
    "ATOM      2  CB  ALA A   1      11.000  21.000  31.000  1.00 85.00           C\n"
    "ATOM      3  CA  GLY A   2      12.000  22.000  32.000  1.00 70.00           C\n"
    "HETATM    4  O   HOH A 100      50.000  50.000  50.000  1.00  0.00           O\n"
)
coords = parse_pdb_coordinates(dummy_pdb, "CA")
check("extracts only CA atoms", coords.shape == (2, 3), f"-> shape={coords.shape}")
check("first coord correct", list(coords[0]) == [10.0, 20.0, 30.0])

coords_all = parse_pdb_coordinates(dummy_pdb, "CB")
check("extracts CB atoms", coords_all.shape == (1, 3))

coords_empty = parse_pdb_coordinates("", "CA")
check("empty pdb returns empty array", coords_empty.shape[0] == 0, f"-> {coords_empty.shape}")

# ─────────────────────────────────────────────
print("\n[8] calculate_rmsd")
c1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
c2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
rmsd = calculate_rmsd(c1, c2)
check("identical coords -> RMSD=0", rmsd == 0.0, f"-> {rmsd}")

c3 = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
rmsd2 = calculate_rmsd(c1, c3)
expected = float(np.sqrt(np.mean([1.0, 1.0])))
check("shifted coords RMSD correct", abs(rmsd2 - expected) < 1e-9, f"-> {rmsd2:.4f} vs {expected:.4f}")

# length mismatch
c_short = np.array([[0.0, 0.0, 0.0]])
rmsd3 = calculate_rmsd(c1, c_short)
check("length mismatch -> None", rmsd3 is None)

# empty arrays
rmsd4 = calculate_rmsd(np.array([]), np.array([]))
check("empty arrays -> None", rmsd4 is None)

# ─────────────────────────────────────────────
print("\n[9] extract_plddt_from_pdb")
pdb_with_plddt = (
    "ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 85.50           C\n"
    "ATOM      2  N   ALA A   1      10.500  20.500  30.500  1.00 85.50           N\n"
    "ATOM      3  CA  GLY A   2      12.000  22.000  32.000  1.00 72.30           C\n"
    "ATOM      4  CA  VAL A   3      14.000  24.000  34.000  1.00 45.10           C\n"
)
res, plddt = extract_plddt_from_pdb(pdb_with_plddt)
check("extracts 3 CA atoms (not N)", len(plddt) == 3, f"-> {len(plddt)}")
check("correct residue numbers", res == [1, 2, 3], f"-> {res}")
check("correct pLDDT values", abs(plddt[0] - 85.5) < 0.01 and abs(plddt[1] - 72.3) < 0.01, f"-> {plddt}")

# ─────────────────────────────────────────────
print("\n[10] analyze_sequence (GFP sequence)")
gfp = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
analysis = analyze_sequence(gfp)
check("returns dict (not None)", analysis is not None)
if analysis:
    check("has 'length' key", "length" in analysis)
    check("length correct", analysis["length"] == len(gfp), f"-> {analysis['length']} vs {len(gfp)}")
    check("has molecular_weight", "molecular_weight" in analysis)
    check("MW is positive float", analysis["molecular_weight"] > 0)
    check("has aromaticity", "aromaticity" in analysis)
    check("has instability_index", "instability_index" in analysis)
    check("has isoelectric_point", "isoelectric_point" in analysis)
    check("has aa_composition", "aa_composition" in analysis)
    check("aa_composition is dict", isinstance(analysis["aa_composition"], dict))
    check("has secondary_structure", "secondary_structure" in analysis)
    ss = analysis["secondary_structure"]
    check("secondary_structure has helix", "helix" in ss)
    check("secondary_structure has turn", "turn" in ss)
    check("secondary_structure has sheet", "sheet" in ss)

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} checks failed:")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED!")
print("=" * 60)
