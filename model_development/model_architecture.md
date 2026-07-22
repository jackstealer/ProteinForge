# AI/ML Model Development

ProteinForge utilizes a dual-model architecture, leveraging two of the most advanced AI frameworks in modern structural biology: **ESMFold** and **AlphaFold**. Rather than training a model from scratch—which requires millions of GPU hours—we focus on the integration, hyperparameter tuning, and downstream application of these models.

## 1. ESMFold (Evolutionary Scale Modeling)
ESMFold is a protein structure prediction model built by Meta AI. It is unique because it relies on a **protein language model** rather than Multiple Sequence Alignments (MSAs).

### Architecture
- **Input:** Raw amino acid sequences (e.g., "MTEITAAMVKELRESTGA...").
- **Embedding:** The sequence is passed through an LLM trained on millions of evolutionary sequences (ESM-2), learning the hidden "grammar" of biology.
- **Folding Head:** The embeddings are fed into a folding module that outputs 3D atomic coordinates.
- **Why we use it:** Because it skips the MSA generation step, ESMFold is incredibly fast, returning structures in 30-60 seconds via the ESM Metagenomic Atlas API. It is used for *novel* sequences.

## 2. AlphaFold 2 (DeepMind)
AlphaFold 2 solved the protein folding problem by achieving experimental-level accuracy.

### Architecture
- **Input:** Amino acid sequence + MSAs.
- **Evoformer Blocks:** A deep neural network architecture that iteratively updates representations of the sequence and pairwise distances between amino acids.
- **Why we use it:** We integrate with the AlphaFold EBI Database to fetch pre-computed structures for *known* proteins (via UniProt ID). This acts as our "gold standard" reference point.

## Bias-Variance Tradeoff & Hyperparameter Tuning
In this project, model "development" also includes tuning how we analyze and filter the AI outputs. We expose 6 hyperparameters to the user to manage the bias-variance tradeoff:

1. **pLDDT Confidence Threshold:** Determines the cutoff for filtering out uncertain regions (lowering variance, but increasing bias by potentially cutting true flexible regions).
2. **Kyte-Doolittle Window Size:** Adjusts the smoothing of hydrophobicity calculations.
3. **Disorder Threshold:** Adjusts the sensitivity of the IUPred-based intrinsic disorder predictions.
4. **Gaussian Smoothing Sigma:** Controls the B-factor normalization applied to the 3D viewer.
