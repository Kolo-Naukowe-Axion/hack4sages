# Task 6: Models Page

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-5 must be complete.

Build the entire models page, then review.

## Files

- Create: `web/src/components/ModelCard.tsx`
- Create: `web/src/components/ArchitectureFlow.tsx`
- Create: `web/src/components/ComparisonTable.tsx`
- Create: `web/src/components/Footer.tsx`
- Modify: `web/src/app/models/page.tsx`

## Section A: Page Header

- "Models" — `font-display text-4xl text-heading`
- "Three approaches to biosignature detection — quantum and classical, compared side by side." — `text-muted max-w-2xl`
- Left-aligned, `py-12 px-8`

## Section B: Model Detail Cards (stacked, min-h-[60vh] total)

### ArchitectureFlow component

Horizontal CSS flow diagram. Props: `steps: { label: string; type: "quantum" | "classical" }[]`
- Each step: rounded pill `px-4 py-2 rounded-full text-sm font-mono`
- Quantum: `bg-cyan/10 border border-cyan/30 text-cyan`
- Classical: `bg-surface border border-border text-muted`
- Arrow separators: Lucide ChevronRight in `text-border`
- Flex-wrap on mobile

### ModelCard component

Frosted glass with expand/collapse (Lucide ChevronDown/ChevronUp):
- Header: title + type badge + toggle icon
- Body: description + ArchitectureFlow + stats table
- Stats rows: Accuracy, Precision, Recall, Training Time, Qubits, Hardware
- Realistic values mixed with "—" (not all dashes)

### 3 cards, stacked with `gap-6`:

**1. QELM — Vetrano Architecture**
- "Reproduction of Vetrano et al. 2025 — the first quantum atmospheric retrieval on real hardware. Angle encoding into a random quantum reservoir, Z-basis measurement, linear output via SVD."
- Flow: Spectrum → PCA → RX Encoding → RY+CNOT Reservoir → Z-Measurement → SVD → Output
- Quantum steps: RX Encoding, RY+CNOT Reservoir, Z-Measurement

**2. QELM — Extended Topology**
- "Our modified circuit with different entanglement pattern and adjusted depth. Tests whether topology affects classification performance on biosignature data."
- Flow: Spectrum → PCA → Angle Encoding → Modified Reservoir → Measurement → SVD → Output
- Quantum steps: Angle Encoding, Modified Reservoir, Measurement

**3. Classical Baseline**
- "The strongest traditional ML model — Random Forest, XGBoost, or CNN. Maximum hyperparameter optimization. The benchmark everything else is measured against."
- Flow: Spectrum → PCA → Feature Engineering → Ensemble/CNN → Output
- All classical steps

## Section C: Comparison Table (min-h-[40vh])

Full-width frosted glass table:
- Columns: Metric | QELM Vetrano | QELM Extended | Classical
- Rows: Accuracy, Precision, Recall, F1, Training Time, Inference Time, Hardware, Qubits
- Values: e.g. Accuracy 94.2% | 91.7% | 96.8%, Hardware: Odra 5 | VTT Q50 | GPU, Qubits: 5 | 53 | —
- Header: `bg-surface/50 font-display text-heading`
- Quantum columns: `border-l-2 border-cyan/20`
- Alternating rows: transparent / `bg-surface/10`
- Values: `font-mono text-center`

## Section D: Footer (py-20)

- Gradient separator
- Asymmetric 3-column layout:
  - Col 1: "About ExoBiome" + 2-3 sentences real project motivation
  - Col 2: "Research" + link to Vetrano et al. 2025 (arXiv:2509.03617) + MultiREx paper link
  - Col 3: "Team" + 4 member slots + "HACK-4-SAGES 2026"
- Bottom bar: hackathon badge `bg-surface/30 rounded-xl px-6 py-3` + tech stack list
- All `text-muted text-sm`

## Review

Playwright screenshot of full page. Check:
- [ ] Model cards feel substantial, not thin
- [ ] Architecture flow readable, quantum steps visually distinct
- [ ] Comparison table looks professional
- [ ] Footer is clean and balanced
- [ ] Expand/collapse works

## Commit

```bash
git add web/src/ && git commit -m "feat: models page with detail cards, comparison table, footer"
```
