"""
Test script to verify ProteinForge installation
Run this before deploying to check all dependencies
"""
import sys

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("  ✅ streamlit")
    except ImportError as e:
        print(f"  ❌ streamlit: {e}")
        return False
    
    try:
        import py3Dmol
        print("  ✅ py3Dmol")
    except ImportError as e:
        print(f"  ❌ py3Dmol: {e}")
        return False
    
    try:
        from stmol import showmol
        print("  ✅ stmol")
    except ImportError as e:
        print(f"  ❌ stmol: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ pandas")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
        return False
    
    try:
        import plotly
        print("  ✅ plotly")
    except ImportError as e:
        print(f"  ❌ plotly: {e}")
        return False
    
    try:
        from Bio import SeqIO
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        print("  ✅ biopython")
    except ImportError as e:
        print(f"  ❌ biopython: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
    
    try:
        import requests
        print("  ✅ requests")
    except ImportError as e:
        print(f"  ❌ requests: {e}")
        return False
    
    return True


def test_utils():
    """Test utils module."""
    print("\n🔧 Testing utils module...")
    
    try:
        from utils import (
            validate_sequence, parse_fasta, 
            get_example_protein_ids, get_confidence_category
        )
        print("  ✅ utils.py imports successful")
    except ImportError as e:
        print(f"  ❌ utils.py import failed: {e}")
        return False
    
    # Test sequence validation
    is_valid, msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
    if is_valid:
        print("  ✅ Sequence validation works")
    else:
        print(f"  ❌ Sequence validation failed: {msg}")
        return False
    
    # Test invalid sequence
    is_valid, msg = validate_sequence("ABC123")
    if not is_valid:
        print("  ✅ Invalid sequence detection works")
    else:
        print("  ❌ Invalid sequence not detected")
        return False
    
    # Test example proteins
    examples = get_example_protein_ids()
    if len(examples) >= 5:
        print(f"  ✅ Example proteins loaded ({len(examples)} examples)")
    else:
        print("  ❌ Example proteins not loaded properly")
        return False
    
    # Test confidence category
    category = get_confidence_category(95)
    if category == "Very High":
        print("  ✅ Confidence categorization works")
    else:
        print("  ❌ Confidence categorization failed")
        return False
    
    return True


def test_files():
    """Test if required files exist."""
    print("\n📁 Testing file structure...")
    
    import os
    
    required_files = [
        'app.py',
        'utils.py',
        'requirements.txt',
        'README.md',
        '.streamlit/config.toml'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} missing")
            all_exist = False
    
    return all_exist


def test_api_connectivity():
    """Test API connectivity (optional)."""
    print("\n🌐 Testing API connectivity...")
    
    import requests
    
    # Test UniProt API
    try:
        response = requests.get("https://rest.uniprot.org/uniprotkb/P69905.fasta", timeout=5)
        if response.status_code == 200:
            print("  ✅ UniProt API accessible")
        else:
            print(f"  ⚠️  UniProt API returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️  UniProt API connection failed: {e}")
    
    # Test AlphaFold DB
    try:
        response = requests.head("https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v4.pdb", timeout=5)
        if response.status_code == 200:
            print("  ✅ AlphaFold DB accessible")
        else:
            print(f"  ⚠️  AlphaFold DB returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️  AlphaFold DB connection failed: {e}")
    
    # Note: ESMFold API test would require POST with sequence, skipping
    print("  ℹ️  ESMFold API test skipped (requires POST request)")
    
    return True  # API tests are non-blocking


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧬 ProteinForge Installation Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import tests failed. Run: pip install -r requirements.txt")
    
    # Test utils
    if not test_utils():
        all_passed = False
        print("\n❌ Utils tests failed. Check utils.py")
    
    # Test files
    if not test_files():
        all_passed = False
        print("\n❌ File structure incomplete")
    
    # Test API connectivity (non-blocking)
    test_api_connectivity()
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Ready to deploy.")
        print("\nRun the app with:")
        print("  streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix issues before deploying.")
        return 1
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
