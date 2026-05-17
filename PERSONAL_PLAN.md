# Personal Execution Plan — Dissertation Completion Guide

**Project**: Noise-Augmented EIT Touch Classification for Prosthetic E-Skin  
**Start**: 13 May 2026  
**Submission deadline**: 29 August 2026, 4pm  
**Presentation submission**: 5 September 2026, 4pm  
**Viva**: Week commencing 8 September 2026  
**Available weeks**: ~15.5 weeks  

---

## Week-by-Week Schedule

### WEEK 1 (May 13–19): Get EIDORS Running

**Goal**: Eliminate the single biggest project risk.

1. **Install EIDORS**
   - Download from http://eidors3d.sourceforge.net/
   - Extract to `matlab/eidors/`
   - Open MATLAB, run `matlab/eidors/startup.m`
   - Verify with: `imdl = mk_common_model('b2c2', 16);` — if no error, EIDORS works

2. **Run your pipeline end-to-end**
   ```matlab
   cd matlab
   main
   ```
   - Expected output: a `.mat` file with 5000 samples (1000 per class)
   - If it crashes: read the error, fix it, try again. Common issues:
     - Path problems → check `addpath` calls in `main.m`
     - EIDORS version mismatch → use EIDORS v3.10+
     - `fwd_solve` errors → check mesh creation parameters

3. **Quick validation**
   - Load the output `.mat` file in Python
   - Plot 10 random voltage vectors per class on the same axes
   - Check: do the classes look different? Are the magnitudes sensible?
   - Run PCA on all samples, colour by class, check visual separation

4. **If EIDORS fails completely**
   - Try the Docker-based EIDORS image if available
   - Try running on a university Linux machine
   - Absolute fallback: use the Bath HPC cluster or ask supervisor for MATLAB access
   - Do NOT spend more than 3 days stuck — escalate to supervisor

**Deliverable**: A working `.mat` dataset file and one PCA plot showing class separation.

---

### WEEK 2 (May 20–26): Fix Code + Get Material Properties

**Goal**: Repair known bugs and ground parameters in physics.

1. **Fix critical Python bugs** (2–3 hours total)

   **Fix 1** — `python/models/cnn1d.py`: Replace brittle `flat_dim` calculation with adaptive pooling:
   ```python
   # Before the FC layers, add:
   self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
   # In forward(), replace flatten logic with:
   x = self.adaptive_pool(x)  # (B, C, 1)
   x = x.view(x.size(0), -1)  # (B, C)
   ```

   **Fix 2** — `python/train.py`: Fix the `--use-noisy` CLI arg:
   ```python
   # Change from:
   parser.add_argument('--use-noisy', default=True, action='store_true')
   # To:
   parser.add_argument('--no-noise', action='store_true', help='Train on clean data')
   # Then use: use_noisy = not args.no_noise
   ```

   **Fix 3** — `python/ablation.py`: Wire in the generated configs:
   ```python
   # In run_ablation(), after the 4 hardcoded experiments, add:
   configs = generate_ablation_configs()
   for config in configs:
       # Run experiment with this noise config
       ...
   ```

   **Fix 4** — Load `config.yaml` in train.py and evaluate.py:
   ```python
   import yaml
   with open('python/configs/config.yaml') as f:
       cfg = yaml.safe_load(f)
   ```

2. **Obtain material properties** (ongoing this week)
   
   What you need from the e-skin material:
   - **Young's modulus** (E, in kPa) — how stiff is it?
   - **Piezoresistive gauge factor** (k or GF) — how much does conductivity change per unit strain?
   - **Baseline conductivity** (σ₀, in S/m) — what is the resting conductivity?
   - **Poisson's ratio** (ν) — for stress/strain conversion
   
   Where to get these:
   - Option A: Ask whoever is providing the material (Bath lab contact)
   - Option B: Find a published paper with full material characterisation of the same or similar material. Good candidates:
     - Paper [19] (variable sensitivity hydrogel + carbon-black elastomer)
     - Paper [3]/[5] (multifunctional hydrogel)
     - Paper [4] (ultra-stretchable hydrogel)
   - Option C: Use datasheet values from commercial piezoresistive elastomers (e.g., Wacker Elastosil, Zoflex)

3. **Build the force-to-conductivity mapping**
   
   Once you have the material properties, the conversion is:
   
   ```
   Strain (ε) = Force / (Area × E)
   Δσ = σ₀ × GF × ε
   ```
   
   For each class, use CoST force data to define F and A, then compute Δσ:
   
   | Class | Force (N) | Area (cm²) | ε (using your E) | Δσ (using your GF) |
   |---|---|---|---|---|
   | No contact | 0 | 0 | 0 | 0 |
   | Light touch | 0.5–3 | 1–3 | calculate | calculate |
   | Firm press | 5–15 | 2–4 | calculate | calculate |
   | Point contact | 2–5 | 0.5–1 | calculate | calculate |
   | Distributed | 1–5 | 8–15 | calculate | calculate |
   
   Update `generate_sample.m` with the calculated σ and radius ranges.

**Deliverable**: All 4 code fixes committed. Material properties documented. Updated class parameters.

---

### WEEK 3 (May 27–June 2): Validate Classes + Generate Pilot Dataset

**Goal**: Confirm the 5 classes are separable and noise parameters are reasonable.

1. **Generate 500 samples per class (2,500 total) — clean only**
   - Modify `main.m` to generate 500 per class temporarily
   - Run and save as `pilot_clean.mat`

2. **Visualise class separability**
   - PCA (2D scatter, coloured by class)
   - t-SNE (perplexity=30, coloured by class)
   - Per-class voltage magnitude histograms
   - If classes overlap badly → increase Δσ gaps or adjust radius ranges

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

### WEEK 4 (June 3–9): Full Dataset Generation

**Goal**: Produce the final dataset for all experiments.

1. **Generate full dataset**: 5,000 samples per class × 5 classes = 25,000 samples
   - Estimated time: 1–4 hours depending on machine (forward solve × 25,000)
   - If too slow: use `parfor` in MATLAB or reduce to 3,000 per class (15,000 total — still fine)
   - Save as `dataset_full.mat` (v7.3 for >2GB support)

2. **Split into train/val/test**
   - Run `load_dataset.py` to verify it loads correctly
   - Check: 70/15/15 split, stratified, reproducible with seed
   - Confirm label encoding is correct (0-indexed in Python)

3. **Compute dataset statistics**
   - Per-class sample counts (should be balanced)
   - Feature ranges (min, max, mean, std per measurement index)
   - Save stats for the Results chapter

4. **Back up the dataset**
   - Copy to a second location (USB, OneDrive, whatever)
   - This file is irreplaceable without re-running EIDORS for hours

**Deliverable**: `dataset_full.mat` generated, loaded in Python, split verified, backed up.

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

**Goal**: Isolate which noise components matter most.

1. **Single-component ablation**
   - Train CNN with ONLY Gaussian noise → evaluate
   - Train CNN with ONLY contact impedance → evaluate
   - Train CNN with ONLY drift → evaluate
   - Train CNN with ONLY electrode bias → evaluate
   - Train CNN with ONLY quantisation → evaluate
   
   This gives 5 experiments, each showing the individual effect.

2. **Leave-one-out ablation**
   - Train CNN with ALL noise EXCEPT Gaussian → evaluate
   - Train CNN with ALL noise EXCEPT contact impedance → evaluate
   - ... (5 more experiments)
   
   This shows: "what happens if you remove one component?"

3. **Full augmentation (all 5 components)**
   - Already done in Week 6, but confirm result

4. **Create ablation summary table**

   | Training noise config | Accuracy (noisy test) | Macro-F1 (noisy test) | Δ vs clean-trained |
   |---|---|---|---|
   | None (clean) | X% | X | baseline |
   | Gaussian only | X% | X | +Y% |
   | Contact impedance only | X% | X | +Y% |
   | ... | ... | ... | ... |
   | All combined | X% | X | +Y% |

5. **Interpret**
   - Rank components by impact
   - Which single component gives the most robustness?
   - Does combining all always beat individual components?

**Deliverable**: Ablation table with 12+ rows. Ablation heatmap figure. Clear ranking of noise components.

---

### WEEK 8 (July 1–7): Robustness Severity Sweep

**Goal**: Test whether noise-trained models generalise beyond training severity.

1. **Define severity levels**
   - 0.5×, 1×, 1.5×, 2×, 2.5×, 3× the default noise parameters
   - For each level, scale ALL noise component magnitudes proportionally

2. **Evaluate all models at each severity**
   - Clean-trained CNN at 6 severity levels
   - Noise-trained CNN at 6 severity levels
   - Best baseline (from Week 5) at 6 severity levels

3. **Plot robustness curves**
   - X-axis: noise severity multiplier
   - Y-axis: accuracy (or macro-F1)
   - Lines: one per model/training condition
   - Expected: clean-trained collapses at 1.5×+, noise-trained degrades gracefully

4. **Find the crossover point**
   - At what severity does clean-trained fall below 50% (random for 5 classes = 20%)?
   - At what severity does noise-trained fall below 80%?
   - The gap between these is your "robustness margin"

5. **Statistical significance at key severity levels**
   - At 2× and 3×: is the difference between clean-trained and noise-trained significant?
   - Paired t-test on 5-fold results

**Deliverable**: Robustness curve figure (publication quality). Severity analysis table. Key p-values.

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
   - 3.4 Noise model (600 words) — for each of 5 components: physical source, equation, parameter value, citation. Include the key equations:
     - Gaussian: $v_{\text{noisy}} = v + \mathcal{N}(0, \sigma_n^2)$ where $\sigma_n = \|v\| / (10^{\text{SNR}/20})$
     - Contact impedance: $v_{\text{noisy}} = v \cdot z_i$, $z_i \sim \text{LogNormal}(0, \sigma_z^2)$
     - etc.
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

## Material Properties: What to Do When You Get Them

When you receive the e-skin material properties, do the following:

### 1. Record these values

| Property | Symbol | Unit | Value |
|---|---|---|---|
| Young's modulus | E | kPa | ? |
| Gauge factor | GF | dimensionless | ? |
| Baseline conductivity | σ₀ | S/m | ? |
| Poisson's ratio | ν | dimensionless | ? |
| Thickness | t | mm | ? |

### 2. Compute force-to-conductivity mapping

```matlab
% For each class, given Force (N) and Area (m²):
strain = Force / (Area * E * 1000);  % E in kPa → Pa
delta_sigma = sigma_0 * GF * strain;
conductivity = sigma_0 + delta_sigma;
```

### 3. Update generate_sample.m

Replace the current hardcoded ranges with computed values:

```matlab
case 'light'
    % Force: 0.5–3 N, Area: 1–3 cm² (from CoST)
    % Using E = [your value] kPa, GF = [your value]
    radius = [computed from area range];
    conductivity = [computed from force/area/material];
```

### 4. Document the chain

In your Methodology chapter, write:
> "Contact force ranges were derived from empirical measurements in the Corpus of Social Touch (Jung et al., 2015). These forces were converted to conductivity changes using the piezoresistive model of [material source], with Young's modulus E = X kPa and gauge factor GF = Y, yielding Δσ ranges of [values] for each class."

### 5. If properties arrive AFTER you've already generated data

That's fine. Regenerate the dataset with updated parameters (takes a few hours). Update all results. The methodology is stronger with material-grounded parameters, so it's worth regenerating even late in the process.

---

## Daily Habits That Will Keep You On Track

1. **Start each day** by opening `results/` and checking what's there vs what's needed.
2. **End each day** by committing code changes to git with a message describing what was done.
3. **Weekly checkpoint** (every Sunday): update a one-line status per week in this document.
4. **If stuck for >2 hours** on a technical problem: write down what you've tried, then move to a different task. Return with fresh eyes the next day.
5. **Write as you go** — don't leave all writing to the end. Even 200 words per day = 2,800 words per fortnight.

---

## Emergency Fallback Positions

| Problem | Fallback |
|---|---|
| EIDORS never works | Use a simplified analytical EIT model (circular homogeneous with known sensitivity matrix). Less realistic but still valid for the noise robustness question. |
| Classes not separable | Reduce to 3 classes (no contact / light / firm). Still publishable. |
| CNN doesn't outperform baselines | That IS a result. Report it honestly. "Noise augmentation benefits all model families equally" is a valid finding. |
| Material properties never arrive | Use published values from paper [19] or [3]. State this explicitly as an assumption. |
| Timeline slips badly | Drop ablation to single-component only (5 experiments instead of 12+). Drop severity sweep to 3 points instead of 6. |
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
| 1 | May 13–19 | EIDORS running, pilot data | |
| 2 | May 20–26 | Code fixed, material props, class params | |
| 3 | May 27–Jun 2 | Validated classes, pilot visualisations | |
| 4 | Jun 3–9 | Full dataset generated | |
| 5 | Jun 10–16 | Baselines trained and evaluated | |
| 6 | Jun 17–23 | CNN trained, statistical tests done | |
| 7 | Jun 24–30 | Ablation study complete | |
| 8 | Jul 1–7 | Severity sweep complete | |
| 9 | Jul 8–14 | Buffer / bonus analyses | |
| 10 | Jul 15–21 | Methodology + Results drafted | |
| 11 | Jul 22–28 | Literature Review drafted | |
| 12 | Jul 29–Aug 4 | Intro + Discussion + Conclusion drafted | |
| 13 | Aug 5–11 | Full draft assembled and self-reviewed | |
| 14 | Aug 12–18 | Supervisor feedback received | |
| 15 | Aug 19–25 | Final revisions complete | |
| 16 | Aug 26–29 | SUBMITTED | |
