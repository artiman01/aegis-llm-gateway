# ADR 0003: Isotropic ZCA-Whitening and Lexical Saliency Gate for Semantic Caching

- **Status:** Accepted
- **Date:** 2026-10-02
- **Author:** Principal Software Architect
- **Deciders:** Core Engineering Team

---

## 1. Context and Problem Statement

In ADR-0002, AegisLLM established a two-tier caching architecture featuring an L2 Semantic Cache powered by dense embeddings (FastEmbed `all-MiniLM-L6-v2`, 384 dimensions) and cosine similarity. In production deployments, standard cosine similarity on raw transformer embeddings exhibits two severe algorithmic pathologies:

### 1.1 The Anisotropy / Representation Degeneration Problem ("The Cone Effect")
Empirical research (Ethayarajh 2019, Gao et al. 2019, Su et al. 2021) demonstrates that contextualized language model representations suffer from systemic geometric anisotropy. Rather than spanning the entire embedding space $\mathbb{R}^d$, sentence embeddings collapse into a narrow, high-density cone oriented around a dominant directional offset.

Consequently:
- Arbitrary, semantically unrelated queries exhibit an unnaturally high baseline cosine similarity (frequently between $0.65$ and $0.85$).
- The effective dynamic range for thresholding is compressed into a narrow band (e.g., $[0.88, 0.95]$), making discrimination between true duplicates and conceptual divergence brittle and sensitive to calibration errors.

### 1.2 Temporal and Entity Drift (The "Numerical/Identifier" Blind Spot)
Transformer attention mechanisms distribute probability mass across all tokens in a prompt. In structured or fact-seeking queries, grammatical scaffolding constitutes 85–95% of sentence tokens:
- **Prompt A:** *"What was the corporate earnings report for Q2 2021?"*
- **Prompt B:** *"What was the corporate earnings report for Q2 2024?"*

Because 9 out of 10 tokens match verbatim, the raw cosine similarity exceeds $0.94$. Standard semantic caches return a **False-Positive Cache Hit**, serving stale 2021 earnings data for a 2024 query, triggering silent hallucinations and compliance violations.

---

## 2. Decision: Two-Phase Isotropic Whitened Saliency Gate

We adopt an advanced two-phase semantic verification pipeline integrating **Zero-phase Component Analysis (ZCA) Whitening** and a deterministic **Lexical Saliency Invariant Gate**.

```
Query Prompt
     │
     ▼
┌─────────────────────────────────┐
│ Dense Embedding Extraction      │ FastEmbed (384-dim)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ ZCA-Whitening Transformation    │ Centering & Covariance Diagonalization:
│ W = U · Λ^(-1/2) · U^T          │ W · (x - μ) -> Unit Hypersphere S^(d-1)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Phase 1: Whitened Cosine Metric │ High dynamic range, isotropic distance
│ sim(x_w, y_w) >= Threshold?     │
└────────────────┬────────────────┘
                 │ PASSED (sim >= threshold)
                 ▼
┌─────────────────────────────────┐
│ Phase 2: Lexical Saliency Gate  │ Regex Invariant Extractor:
│ Numbers, Dates, IDs, Entities   │ Numerics, ISO Dates, UUIDs, Named Tokens
└────────────────┬────────────────┘
                 │
      ┌──────────┴──────────┐
      │ Saliency Match?     │
      ▼ YES                 ▼ NO (Temporal/Entity Mismatch)
┌───────────┐         ┌───────────┐
│ CACHE HIT │         │ CACHE MISS│ (Force upstream LLM evaluation)
└───────────┘         └───────────┘
```

---

## 3. Mathematical Foundations of ZCA-Whitening

Given a calibration set of $N$ embedding vectors $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, the distribution exhibits sample mean $\mu \in \mathbb{R}^d$ and sample covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$:

$$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$

$$\Sigma = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T + \epsilon I_d$$

Where $\epsilon = 10^{-5}$ is a Tikhonov regularization parameter ensuring strict positive-definiteness and numerical stability.

Applying Singular Value Decomposition (SVD) / Eigendecomposition to $\Sigma$:

$$\Sigma = U \Lambda U^T$$

Where:
- $U \in \mathbb{R}^{d \times d}$ is the orthogonal matrix of eigenvectors ($U^T U = I_d$).
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$ contains sorted eigenvalues ($\lambda_i > 0$).

### 3.1 ZCA Whitening Matrix Derivation
Unlike Principal Component Analysis (PCA) whitening which rotates vectors into unaligned principal axes, **Zero-phase Component Analysis (ZCA) Whitening** applies a symmetric transformation that aligns the whitened vectors as closely as possible to the original feature coordinates in terms of minimal mean squared error:

$$W_{\text{ZCA}} = U \Lambda^{-1/2} U^T$$

Where:

$$\Lambda^{-1/2} = \text{diag}\left(\frac{1}{\sqrt{\lambda_1 + \epsilon}}, \frac{1}{\sqrt{\lambda_2 + \epsilon}}, \dots, \frac{1}{\sqrt{\lambda_d + \epsilon}}\right)$$

For an arbitrary input vector $x \in \mathbb{R}^d$, the transformed isotropic vector $\tilde{x}$ and its spherical projection $\hat{x}$ are:

$$\tilde{x} = W_{\text{ZCA}}(x - \mu)$$

$$\hat{x} = \frac{\tilde{x}}{\|\tilde{x}\|_2}$$

### 3.2 Theoretical Guarantees
1. **Zero Mean:** $\mathbb{E}[\tilde{x}] = \mathbf{0}$.
2. **Identity Covariance:** $\text{Cov}(\tilde{x}) = W \Sigma W^T = (U \Lambda^{-1/2} U^T)(U \Lambda U^T)(U \Lambda^{-1/2} U^T) = I_d$.
3. **Hypersphere Isotropicity:** The degenerate "cone" is expanded uniformly over the unit hypersphere $S^{d-1}$, maximizing the variance of informational components and nullifying background offset bias.

---

## 4. Phase 2: Deterministic Lexical Saliency Gate

Even in an isotropic embedding space, high sentence similarity may persist when queries differ solely by a critical entity or temporal discriminator. To guarantee zero false-positive contamination:

1. **Invariant Extraction:** A high-speed regular-expression pipeline extracts all invariant lexical tokens from both the incoming query prompt $Q$ and candidate cached prompt $C$:
   - **Numerics:** Integers, floats, currencies (`\b\d+(?:\.\d+)?\b`).
   - **Temporal Invariants:** Years, quarters, ISO dates (`\b(?:19|20)\d{2}\b`, `Q[1-4]`, `\b\d{4}-\d{2}-\d{2}\b`).
   - **Key Identifiers:** Alphanumeric IDs, hashes, account numbers (`\b[A-Za-z]+-\d+\b`, `ID-\w+`).
   - **Capitalized Named Entities:** Distinct proper nouns (`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b`).

2. **Saliency Gate Decision Invariant:**
   Let $S(P)$ denote the multiset of extracted salient invariants for prompt $P$.
   
   $$\text{GateMatch}(Q, C) = \begin{cases} 
   \text{True}, & \text{if } S(Q) = S(C) \\
   \text{False}, & \text{if } S(Q) \neq S(C)
   \end{cases}$$

If $S(Q) \neq S(C)$ (for instance, $S(Q)$ contains `"2024"` and $S(C)$ contains `"2020"`), the gate **forces a CACHE MISS**, irrespective of how close the cosine similarity is to $1.0$.

---

## 5. Consequences and Operational Trade-offs

### Positive Consequences
- **Elimination of Semantic False Hits:** Resolves the subtle entity drift problem without requiring larger or slower embedding models.
- **Wider Dynamic Range:** ZCA whitening allows calibrating similarity thresholds reliably across $0.80 - 0.95$ without false clustering.
- **Ultra-low Overhead:** ZCA matrix-vector multiplication is an $\mathcal{O}(d^2)$ operation ($384 \times 384 \approx 1.47 \times 10^5$ FLOPs, $< 50\,\mu\text{s}$ using NumPy BLAS). Regex saliency extraction takes $< 30\,\mu\text{s}$.

### Negative Consequences / Mitigation
- **Calibration Requirement:** The ZCA transformation matrix requires pre-calibration on representative corpus vectors or synthetic warm-up prompts.
  - *Mitigation:* The system loads pre-computed calibration weights by default, with an automated warm-up mechanism that falls back to an identity whitening matrix if uncalibrated.
