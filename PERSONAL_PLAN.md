# Personal Execution Plan — Dissertation Completion Guide

**Project**: Noise-Augmented EIT Touch Classification for Prosthetic E-Skin  
**Start**: 13 May 2026  
**Submission deadline**: 29 August 2026, 4pm  
**Presentation submission**: 5 September 2026, 4pm  
**Viva**: Week commencing 8 September 2026  
**Available weeks**: ~15.5 weeks  

---

## Week-by-Week Schedule

### WEEK 3 (May 27–June 2): Validate Classes + Generate Pilot Dataset

**Goal**: Confirm the 5 classes are separable and noise parameters are reasonable.

3. **Generate 2,500 noisy samples and compare**
   - Load `pilot_clean.mat`, check both `dataset_X_clean` and `dataset_X_noisy`
   - Plot clean vs noisy samples side-by-side for each class
   - Check: does noise preserve class structure or destroy it?
   - If noise is too strong at default settings → reduce SNR to 50 dB, reduce drift

4. **Verify noise component toggles**
   - Generate 5 batches, each with only one noise component enabled
   - Compare visually: which component distorts the signal most?
   - This gives you early qualitative ablation insight

5. **Document decisions**
   - Write a short note: "Classes X and Y are well-separated; Z needed adjustment because..."
   - This becomes material for your Methodology chapter

**Deliverable**: Validated class parameters. Confirmed noise levels. Pilot visualisations saved to `results/figures/`.

---

### WEEK 5 (June 10–16): Train Baselines

**Goal**: Establish performance baselines across all conditions.

1. **Train all baselines on clean data**
   ```bash
   python python/train.py --no-noise --model svm
   python python/train.py --no-noise --model rf
   python python/train.py --no-noise --model mlp
   ```
   Record: accuracy, macro-F1, per-class F1

2. **Train all baselines on noisy data**
   ```bash
   python python/train.py --model svm
   python python/train.py --model rf
   python python/train.py --model mlp
   ```

3. **Evaluate all baselines under all 4 conditions**
   - Clean→Clean (ceiling)
   - Clean→Noisy (vulnerability)
   - Noisy→Noisy (robustness)
   - Noisy→Clean (generalisation)
   
   Record all results in a CSV or table.

4. **Quick analysis**
   - Which baseline is best on clean data?
   - How much does each degrade under noise?
   - Does noise training help baselines at all?

**Deliverable**: Baseline results table (12 entries: 3 models × 4 conditions). Saved model files.

---

### WEEK 6 (June 17–23): Train CNN + Compare

**Goal**: Train the 1D-CNN and establish whether it outperforms baselines.

1. **Train CNN on clean data**
   ```bash
   python python/train.py --no-noise --model cnn --epochs 100
   ```
   - Monitor training/validation loss curves
   - Save best model checkpoint

2. **Train CNN on noisy data**
   ```bash
   python python/train.py --model cnn --epochs 100
   ```
   - Compare learning curves: does noisy training converge slower? (expected)

3. **Evaluate CNN under all 4 conditions**
   - Same matrix as baselines
   - Record accuracy, macro-F1, confusion matrices

4. **Statistical comparison**
   - Run 5-fold stratified cross-validation for each model × condition
   - Compute mean ± std for accuracy and F1
   - Run paired t-test: CNN-noisy vs CNN-clean on noisy test set
   - Run paired t-test: CNN-noisy vs best-baseline-noisy on noisy test set
   - Record p-values

5. **Generate confusion matrices**
   - 4 matrices for CNN (one per condition)
   - Save as PDF to `results/figures/`

**Deliverable**: CNN results added to comparison table. Statistical tests completed. Confusion matrices saved.

---

### WEEK 7 (June 24–30): Ablation Study

**Goal**: Isolate which noise components matter most using Python-side noise injection.

**Infrastructure (COMPLETED)**: The ablation pipeline now uses Python-side noise injection
via `python/data/noise.py`. No MATLAB dependency at experiment time. The CNN receives fresh
noise realisations each epoch (better generalisation). All experiments are evaluated against
the same full 4-component noise test condition for consistency.

1. **Run full ablation study**
   ```bash
   # Core mismatch (4 experiments: clean→clean, clean→noisy, noisy→noisy, noisy→clean)
   uv run python python/ablation.py --model cnn1d

   # Full per-component ablation (4 single-component + 4 leave-one-out)
   uv run python python/ablation.py --model cnn1d --all-configs

   # Repeat for best baseline
   uv run python python/ablation.py --model random_forest --all-configs
   ```

2. **Single-component ablation (4 experiments)**
   - Train CNN with ONLY Gaussian noise → evaluate
   - Train CNN with ONLY contact impedance → evaluate
   - Train CNN with ONLY electrode bias → evaluate
   - Train CNN with ONLY quantisation → evaluate
   
   Uses `NoiseConfig.only(component)` — each experiment trains on clean data with
   only one noise component applied on-the-fly.

3. **Leave-one-out ablation (4 experiments)**
   - Train CNN with ALL noise EXCEPT Gaussian → evaluate
   - Train CNN with ALL noise EXCEPT contact impedance → evaluate
   - Train CNN with ALL noise EXCEPT electrode bias → evaluate
   - Train CNN with ALL noise EXCEPT quantisation → evaluate
   
   Uses `NoiseConfig.without(component)`.

4. **Create ablation summary table**

   | Training noise config | Accuracy (noisy test) | Macro-F1 (noisy test) | Δ vs clean-trained |
   |---|---|---|---|
   | None (clean) | X% | X | baseline |
   | Gaussian only | X% | X | +Y% |
   | Contact impedance only | X% | X | +Y% |
   | Electrode bias only | X% | X | +Y% |
   | Quantisation only | X% | X | +Y% |
   | Without Gaussian | X% | X | +Y% |
   | Without Contact impedance | X% | X | +Y% |
   | Without Electrode bias | X% | X | +Y% |
   | Without Quantisation | X% | X | +Y% |
   | All combined | X% | X | +Y% |

5. **Interpret**
   - Rank components by impact
   - Which single component gives the most robustness?
   - Does combining all always beat individual components?
   - Compare CNN vs baseline ablation patterns

**Deliverable**: Ablation table with 10+ rows. Ablation heatmap figure. Clear ranking of noise components.

---

### WEEK 8 (July 1–7): Multi-Severity Training + Robustness Sweep

**Goal**: Address over-specialisation and test generalisation beyond training severity.

**Context**: Current results show over-specialisation — the noise-trained CNN achieves 74.1%
at 1.0× severity but drops to 43.3% at 0.5× (worse than expected). This indicates the model
encodes the specific training noise level rather than learning noise-invariant features.

1. **Train multi-severity CNN (domain randomisation)**
   - Enable `noise_augmentation.enabled: true` in `config.yaml`
   - `severity_range: [0.5, 2.0]` samples different noise intensities per batch
   - Each batch sees a different severity → model cannot overfit to one noise level
   ```bash
   uv run python python/train.py --model cnn1d
   ```

2. **Evaluate at all severity levels (7 points)**
   - Severity multipliers: {0.0×, 0.5×, 1.0×, 1.5×, 2.0×, 2.5×, 3.0×}
   - Use `evaluate_severity_sweep_python()` which generates fresh noise at each level
   - Compare: single-severity trained vs multi-severity trained vs clean-trained

3. **Plot robustness curves**
   - X-axis: noise severity multiplier
   - Y-axis: accuracy (or macro-F1)
   - Lines: clean-trained CNN, single-severity CNN, multi-severity CNN, best baseline
   - Expected: multi-severity trained shows flatter curve (less over-specialisation)

4. **Quantify robustness improvements**
   - Compare accuracy at 0.5× (below training level) for single vs multi-severity
   - Compare accuracy at 2.0× and 3.0× (above training level)
   - Calculate degradation slope (accuracy per unit severity increase)
   - Find crossover points where models fall below useful accuracy

5. **Statistical significance at key severity levels**
   - At 0.5×, 1.0×, and 2.0×: is the difference between training regimes significant?
   - Use 5-fold CV results where possible

6. **Investigate BatchNorm contribution to over-specialisation**
   - Consider: does BN encode noise statistics? Would instance normalisation or
     group normalisation produce less severity-specific models?
   - If time permits: quick experiment replacing BN with GN in one comparison

**Deliverable**: Robustness curve figure (publication quality). Multi-severity vs single-severity
comparison. Severity analysis table with degradation slopes. Evidence of over-specialisation
resolved (or characterised).

---

### WEEK 9 (July 8–14): Buffer + Additional Analysis

**Goal**: Catch up on anything delayed. Run bonus analyses if on schedule.

1. **If behind**: Use this week to finish Weeks 7–8 experiments.

2. **If on schedule**, pick 1–2 bonus analyses:
   - **Feature importance**: Which measurement indices contribute most to classification? Use gradient-based saliency on the CNN or RF feature importance.
   - **t-SNE visualisation**: Plot learned CNN features (penultimate layer activations) for clean-trained vs noise-trained models. Does noise training produce tighter clusters?
   - **3D cylindrical mesh attempt**: Try `mk_common_model('b3CR', [16, 2])` in EIDORS. If it works, generate 500 samples and compare voltage patterns to 2D. Report as supplementary validation.
   - **Real data pilot** (if Bath lab access materialised): Even 50 real measurements compared qualitatively to simulated distributions would be transformative.

3. **Finalise all results**
   - All tables in CSV format in `results/tables/`
   - All figures in PDF format in `results/figures/`
   - Ensure all experiments are reproducible from `config.yaml` + saved seeds

**Deliverable**: All experimental work complete. Results directory populated with final outputs.

---

### WEEK 10 (July 15–21): Write Methodology + Results

**Goal**: Draft the two most technical chapters.

1. **Chapter 3: Methodology (2,500 words)**

   Structure:
   - 3.1 Research design overview (200 words) — why simulation-based, why static
   - 3.2 EIT simulation (500 words) — EIDORS, mesh type, electrode config, forward solve. Include mesh figure.
   - 3.3 Touch class definitions (300 words) — table with class, force range (from CoST), Δσ range (from material model), radius range. Justify the force→Δσ conversion.
   - 3.4 Noise model (600 words) — for each of 4 components: physical source, equation, parameter value, citation. Include the key equations:
     - Gaussian: $v_{\text{noisy}} = v + \mathcal{N}(0, \sigma_n^2)$ where $\sigma_n = \|v\| / (10^{\text{SNR}/20})$
     - Contact impedance: $v_{\text{noisy}} = v \cdot z_i$, $z_i \sim \text{LogNormal}(0, \sigma_z^2)$
     - Electrode bias: $v_{\text{noisy}} = v + b_e$, $b_e \sim U(-b_{max}, +b_{max})$
     - Quantisation: $v_{\text{noisy}} = v + q$, $q \sim U(-\text{LSB}/2, +\text{LSB}/2)$
   - 3.5 ML pipeline (500 words) — preprocessing, architectures (table of layers for CNN), training protocol, evaluation protocol
   - 3.6 Ablation design (200 words) — what is toggled, what is measured
   - 3.7 Metrics and statistical testing (200 words) — accuracy, F1, paired t-test, reporting protocol

2. **Chapter 4: Results (2,000 words)**

   Structure:
   - 4.1 Dataset characteristics (200 words) — class balance, feature distributions
   - 4.2 Clean baseline (300 words) — all models on clean data, ceiling performance
   - 4.3 Noise impact on clean-trained models (400 words) — degradation quantified
   - 4.4 Noise-augmented training (400 words) — improvement quantified with p-values
   - 4.5 Ablation findings (400 words) — ranking table, heatmap discussion
   - 4.6 Severity extrapolation (300 words) — robustness curves, crossover points

   Key writing tip: **Lead with the finding, then show the evidence**. Don't describe the table — tell the reader what the table means.

**Deliverable**: Complete drafts of Chapters 3 and 4.

---

### WEEK 11 (July 22–28): Write Literature Review

**Goal**: Synthesise your 72 papers into 2,500 words of thematic analysis.

1. **Structure (follow this exactly)**:
   - 2.1 Electronic skin for prosthetics (400 words) — materials, requirements, biological inspiration. Cite [1], [3], [4], [9], [13], [49], [51]. End with: "most materials work includes limited ML evaluation."
   - 2.2 EIT: principles and limitations (400 words) — forward/inverse problem, ill-posedness, low resolution, EIDORS. Cite [15], [17], [36], [42]. End with: "reconstruction quality depends on noise conditions rarely modelled."
   - 2.3 ML for EIT (500 words) — classical → deep → hybrid. Include comparison table. Cite [14], [19], [20], [21], [22], [26], [27], [64]. End with: "most studies test only under clean or Gaussian-only noise."
   - 2.4 Robustness and sim-to-real (500 words) — THIS IS YOUR KEY SECTION. Cite [33], [38], [39], [40], [43], [44], [46], [47], [48]. End with: "no study systematically ablates multiple physically-motivated noise components for EIT tactile classification."
   - 2.5 Touch classification (400 words) — static vs temporal, classical vs deep. Cite [6], [54], [56], [57], [60], [64]. End with: "lightweight models with robust training may suit embedded prosthetics."
   - 2.6 Ethical, economic, sustainability (300 words) — safety, clinical trust, cost of simulation-first, e-waste. Cite [2], [8], [13].

2. **Writing rules for the lit review**:
   - Every paragraph must cite ≥2 sources
   - Never summarise a paper in isolation — always compare, contrast, or synthesise
   - End every subsection with a gap statement
   - Use present tense for established knowledge, past tense for specific findings
   - Include ONE comparison table (method × data × validation × noise tested)

3. **Comparison table template**:

   | Study | Method | Data | Validation | Noise? | Key limitation |
   |---|---|---|---|---|---|
   | Chen et al. (2023) | CNN-TR | Sim+Real | Lab test | Gaussian only | No ablation |
   | Paper [19] | FFNN | Real | 80/10/10 | No | Small dataset |
   | Paper [20] | FISTA | Real | MIoU | No | Conductive objects only |
   | Paper [26] | V²A-Net | Sim | SSIM=0.92 | Partial | No classification |
   | Paper [64] | LogitBoost | Real | 5-fold | No | Limited features |
   | **This work** | **1D-CNN** | **Sim** | **5-fold + sweep** | **5-component** | **No real data** |

**Deliverable**: Complete draft of Chapter 2. Comparison table finalised.

---

### WEEK 12 (July 29–August 4): Write Introduction + Discussion + Conclusion

**Goal**: Complete all remaining chapters.

1. **Chapter 1: Introduction (900 words)**
   - 1.1 Context (200 words): prosthetic e-skin need, human touch complexity
   - 1.2 EIT for e-skin (200 words): why it's attractive, why it's hard
   - 1.3 The robustness gap (200 words): clean training → real-world failure
   - 1.4 Contributions (150 words): three numbered bullet points
   - 1.5 Structure (150 words): chapter guide

2. **Chapter 5: Discussion (1,250 words)**
   - 5.1 Interpretation (400 words): what the results mean for EIT deployment
   - 5.2 Comparison with literature (300 words): explicitly reference your comparison table. "Our noise-trained CNN achieves X% under Y conditions, compared to Z% reported by [author] under clean conditions. This supports the claim that..."
   - 5.3 Limitations (300 words): be specific and honest:
     - Simulation only (no hardware proof)
     - 2D simplification (real arms are 3D)
     - Static contact only (no dynamics)
     - Material model assumption (specific GF used)
     - 5 classes (real interaction is more varied)
   - 5.4 Broader implications (250 words): ethical deployment, economic value of simulation-first, sustainability of reducing hardware iterations

3. **Chapter 6: Conclusion (500 words)**
   - Summarise contributions (150 words)
   - Answer each RQ explicitly (200 words):
     - RQ1: "Noise-augmented training improved accuracy by X% under degraded conditions (p < 0.01)."
     - RQ2: "Contact impedance variation caused the greatest single-component degradation (Y% drop), followed by drift (Z%)."
     - RQ3: "The noise-trained CNN maintained >W% accuracy at 2× training severity, compared to V% for the clean-trained model."
   - Future work (150 words): real-data validation, 3D mesh, temporal models, domain adaptation, EIT-C benchmark

4. **Abstract (250 words)** — write this LAST, after everything else is done

**Deliverable**: Complete first draft of all chapters.

---

### WEEK 13 (August 5–11): Integrate, Polish, First Complete Draft

**Goal**: Assemble the full document and do a self-review.

1. **Assemble all chapters** into one document (Word or LaTeX)
2. **Format check**:
   - Arial 11pt, 1.5 spacing
   - 2.5cm margins (3cm left)
   - Centre-bottom page numbering
   - Harvard referencing throughout
3. **Add front matter**:
   - Title page
   - Contents list
   - Abstract
   - List of figures/tables
   - Acknowledgements
4. **Self-review checklist**:
   - [ ] Every figure has a caption and is referenced in text
   - [ ] Every table has a title and is referenced in text
   - [ ] Every claim has a citation or is supported by your results
   - [ ] Research questions are stated in Ch1 and answered in Ch6
   - [ ] Ethical/sustainability discussion is present
   - [ ] Word count is ≤10,000 (excl. references and appendices)
   - [ ] All references use Harvard format consistently
5. **Run a word count** — cut ruthlessly if over

**Deliverable**: Complete first draft ready for supervisor.

---

### WEEK 14 (August 12–18): Supervisor Feedback

**Goal**: Get and integrate feedback.

1. **Send draft to supervisor** at the start of this week
2. While waiting:
   - Clean up GitHub repository
   - Write `README.md` with installation and reproduction instructions
   - Ensure all experiments can be re-run from the saved configs
   - Prepare appendices (additional figures, full ablation tables, code documentation)
3. **When feedback arrives**:
   - Prioritise structural/argument feedback over typos
   - If major issues flagged → fix immediately
   - If minor → batch fixes in Week 15

**Deliverable**: Feedback received. Priority fix list created.

---

### WEEK 15 (August 19–25): Final Revisions

**Goal**: Implement all feedback and polish to submission quality.

1. **Address all supervisor comments**
2. **Proofread** entire document (read aloud if possible)
3. **Check all references** — every in-text citation has a full reference entry and vice versa
4. **Verify figures** print well in black-and-white (examiners may print)
5. **Final formatting pass**:
   - Consistent heading styles
   - No orphan headings (heading at bottom of page with content on next)
   - Tables don't split across pages unnecessarily
   - Page numbers present and correct in contents
6. **Export as PDF** — check nothing broke in conversion

**Deliverable**: Final dissertation PDF ready for submission.

---

### WEEK 16 (August 26–29): Submit

**Goal**: Submit before 4pm Friday 29 August.

1. **Monday–Wednesday**: Final read-through. Fix any last issues.
2. **Thursday**: Upload to Moodle submission point. Verify upload successful.
3. **Friday 29 August by 4pm**: Deadline. Already submitted. Celebrate.

---

### WEEKS 17–18 (September 1–12): Presentation + Viva

1. **By 5 September 4pm**: Submit recorded PowerPoint voice-over presentation (7 minutes)
   
   Slide structure:
   - Slide 1: Title + name
   - Slide 2: Problem (why this matters)
   - Slide 3: Gap in literature (one sentence)
   - Slide 4: Your approach (noise model + ablation)
   - Slide 5: Key result 1 (robustness improvement)
   - Slide 6: Key result 2 (ablation ranking)
   - Slide 7: Key result 3 (severity extrapolation curve)
   - Slide 8: Limitations + future work
   - Slide 9: Conclusion (contributions)

2. **Week of 8 September**: Viva (13 minutes Q&A)
   
   Prepare for likely questions:
   - "Why didn't you use real data?" → Justify simulation-first, cite cost/access barriers
   - "How do you know your noise model is realistic?" → Cite Adler, Boone, Vilhunen, Kolehmainen
   - "Would this work with more classes?" → Discuss information capacity of 208 measurements
   - "What would you do with another 3 months?" → Real data, 3D mesh, temporal modelling
   - "Why not a more complex model?" → Scientific clarity, deployment constraints, data size


---

## Emergency Fallback Positions

| Problem | Fallback |
|---|---|
| EIDORS never works | Use a simplified analytical EIT model (circular homogeneous with known sensitivity matrix). Less realistic but still valid for the noise robustness question. |
| Classes not separable | Reduce to 3 classes (no contact / light / firm). Still publishable. |
| CNN doesn't outperform baselines | That IS a result. Report it honestly. "Noise augmentation benefits all model families equally" is a valid finding. |
| Material properties never arrive | Use published values from paper [19] or [3]. State this explicitly as an assumption. |
| Timeline slips badly | Drop ablation to single-component only (4 experiments instead of 10+). Drop severity sweep to 3 points instead of 7. |
| Real data access falls through | Expected. Frame as future work. Your simulation contribution stands alone. |

---

## Checklist: Items Required for Distinction (75%+)

- [ ] Clear, bounded research questions stated and answered
- [ ] Thematic literature review with comparison table
- [ ] Physically motivated noise parameters cited from literature
- [ ] Material-grounded class definitions (force → conductivity)
- [ ] Statistical significance on key claims (p-values reported)
- [ ] Full ablation with clear ranking of components
- [ ] Robustness extrapolation beyond training severity
- [ ] Results explicitly tied back to literature in Discussion
- [ ] Ethical/economic/sustainability discussion included
- [ ] Honest limitations section
- [ ] Clean, well-formatted presentation with Harvard referencing
- [ ] Reproducible code repository

---

## Weekly Status Tracker

| Week | Dates | Target | Status |
|---|---|---|---|
| 1 | May 13–19 | EIDORS running, pilot data | ✅ Complete |
| 2 | May 20–26 | Code fixed, material props, class params | ✅ Complete |
| 3 | May 27–Jun 2 | Validated classes, pilot visualisations | 🔄 In progress |
| 4 | Jun 3–9 | Full dataset generated | |
| 5 | Jun 10–16 | Baselines trained and evaluated | ✅ Complete (run_all_experiments.py) |
| 6 | Jun 17–23 | CNN trained, statistical tests done | ✅ Complete (3 seeds × 4 conditions) |
| 7 | Jun 24–30 | Ablation study (Python-side noise) | Ready to run |
| 8 | Jul 1–7 | Multi-severity training + severity sweep | Ready to run |
| 9 | Jul 8–14 | Buffer / bonus analyses | |
| 10 | Jul 15–21 | Methodology + Results drafted | |
| 11 | Jul 22–28 | Literature Review drafted | |
| 12 | Jul 29–Aug 4 | Intro + Discussion + Conclusion drafted | |
| 13 | Aug 5–11 | Full draft assembled and self-reviewed | |
| 14 | Aug 12–18 | Supervisor feedback received | |
| 15 | Aug 19–25 | Final revisions complete | |
| 16 | Aug 26–29 | SUBMITTED | |

---

## Completed Infrastructure (as of Week 3)

The following development work has been completed ahead of schedule:

### Python-Side Noise Augmentation Module (`python/data/noise.py`)
- Full 4-component noise model matching MATLAB `add_noise.m`
- `NoiseConfig` dataclass with factory methods: `.only()`, `.without()`, `.all_off()`, `.from_yaml()`
- Vectorised batch noise application (`apply_noise_batch_vectorised()`) for fast training
- Severity scaling: all parameters scaled by a single multiplier
- Per-electrode structure for contact impedance and bias

### Online Noise Augmentation in Training (`python/train.py`)
- `noise_config` parameter enables on-the-fly noise during CNN training
- `severity_range` parameter samples severity uniformly per batch
- Clean data used as input; noise applied per-batch (different realisation each epoch)
- Addresses over-specialisation by exposing model to multiple noise intensities

### Refactored Ablation Study (`python/ablation.py`)
- Python-side noise injection eliminates MATLAB dependency at experiment time
- Consistent evaluation: all experiments tested against full 4-component noise
- GPU support with proper device handling throughout
- Supports both CNN (on-the-fly noise) and sklearn (pre-applied noise)
- `--all-configs` flag runs 4 single-component + 4 leave-one-out experiments

### Python Severity Sweep (`python/evaluate.py`)
- `evaluate_severity_sweep_python()` generates fresh noise at each severity level
- 7 severity multipliers: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- Computes degradation slope and key severity-level metrics

### Initial Experiment Results (from `run_all_experiments.py`)
- 56 experiments: 4 datasets × 4 models × 4 conditions × 3 seeds
- Best noisy-domain: CNN on raw data (75.1% accuracy, noisy→noisy)
- Best clean-domain: CNN on cleaned data (98.1% accuracy, clean→clean)
- Noise training improvement: +52.5pp for CNN on raw (clean→noisy vs noisy→noisy)
- Over-specialisation identified: CNN drops to 43.3% at 0.5× severity (below training level)
