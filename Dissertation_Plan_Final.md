# Project Proposal

## Title
**Towards Simulation-to-Reality Transfer in EIT Tactile Sensing: A Noise-Augmented Deep Learning Framework for Prosthetic Electronic Skin**

## 1. Executive Summary
This project proposes a robust machine learning framework for classifying tactile interactions from Electrical Impedance Tomography (EIT) measurements in prosthetic electronic skin (e-skin). The core problem is that many EIT learning models are trained on idealised simulation data and then perform poorly under realistic conditions affected by noise, drift, electrode variability, and measurement imperfections.

The proposed solution is to develop a physically motivated, multi-component noise model and integrate it into the training pipeline as structured augmentation. The project will generate synthetic EIT data with EIDORS, train and compare baseline and deep models, and run a systematic ablation study to identify which noise components most affect robustness. The intended contribution is not to claim proven real-world transfer without hardware evidence, but to provide strong empirical evidence that noise-aware training improves resilience under degraded conditions and is a credible pathway towards sim-to-real deployment.

## 2. Problem Statement
Prosthetic systems require reliable tactile perception for safe interaction, object handling, and user confidence. EIT is attractive for e-skin because it supports distributed sensing with fewer interconnects than dense taxel arrays. However, EIT is an ill-posed inverse sensing modality and is highly sensitive to noise and modelling errors. 

Most published pipelines report strong performance under controlled conditions but do not rigorously model realistic corruption processes during training. As a result, models can overfit to idealised measurements and fail when faced with sensor drift, contact impedance variability, or hardware-dependent artefacts.

## 3. Significance and Motivation
This work is significant for three reasons:

1. **Technical relevance**: It addresses a major deployment barrier in EIT-based tactile sensing: robustness under non-ideal measurement conditions.
2. **Clinical and engineering relevance**: More reliable tactile inference could support safer prosthetic control and better user experience.
3. **Research relevance**: It provides a reproducible benchmark-style framework for evaluating corruption robustness in EIT classification.

## 4. Project Aim, Objectives, and Research Questions

### 4.1 Aim
To design and evaluate a noise-augmented machine learning pipeline that improves robustness of EIT-based tactile touch classification for prosthetic e-skin.

### 4.2 Objectives
1. Build a reproducible EIT simulation and data generation pipeline using EIDORS.
2. Define five tactile interaction classes relevant to prosthetic touch sensing.
3. Implement a physically motivated multi-component noise model.
4. Train and compare baseline models (SVM, Random Forest, MLP) and a 1D-CNN.
5. Quantify robustness under controlled corruption severity and unseen degradation levels.
6. Perform ablation to isolate the impact of each noise component and key combinations.
7. Report findings with statistical testing, reproducible code, and publication-ready outputs.

### 4.3 Research Questions
1. How does noise-aware training affect touch classification performance under degraded EIT measurements?
2. Which corruption sources most strongly reduce model robustness?
3. Can noise-augmented models generalise to corruption severity levels beyond those seen during training?

### 4.4 Hypotheses
1. Models trained with physically motivated noise augmentation will outperform clean-trained models on degraded test conditions.
2. Contact impedance variability and drift will degrade performance more than additive Gaussian noise alone.
3. A 1D-CNN with noise-aware training will outperform classical baselines under high corruption severity.

## 5. Proposed Contributions
The project is expected to produce the following contributions:

1. **A physically motivated EIT corruption model** covering Gaussian measurement noise, contact impedance variation, drift, electrode bias, and quantisation effects.
2. **A robust evaluation protocol** for EIT touch classification including clean/noisy cross-condition testing and severity sweeps.
3. **Ablation evidence** identifying which corruption components matter most for performance degradation.
4. **A reproducible open workflow** (MATLAB + Python + tests + documentation) suitable for follow-on research and publication.

## 6. Scope and Boundaries

### In scope
1. Simulation-based EIT data generation with EIDORS.
2. Static touch classification with five classes.
3. Corruption-aware training and robustness evaluation.
4. Baseline and CNN model comparison.

### Out of scope
1. Full real-hardware validation (unless limited access becomes available).
2. Temporal/sliding touch modelling with sequence models.
3. Real-time embedded deployment optimisation.
4. Complete EIT image reconstruction benchmarking as a primary task.

## 7. Methodology

### 7.1 Data generation and simulation
1. Use EIDORS forward modelling to simulate voltage responses for tactile contacts.
2. Start with a validated 2D setup for core experiments.
3. Attempt 3D cylindrical extension as a stretch objective after baseline milestones are complete.

### 7.2 Touch classes
Five classes will be generated and labelled:
1. No contact
2. Light touch
3. Firm press
4. Point contact
5. Distributed contact

### 7.3 Corruption model
Noise/corruption components:
1. Additive Gaussian measurement noise
2. Contact impedance variation
3. Drift-like perturbation
4. Electrode/channel bias
5. Quantisation noise

Each component will be parameterised and toggled independently to support controlled ablation.

### 7.4 Model development
1. Baselines: SVM (RBF), Random Forest, MLP.
2. Primary model: 1D-CNN over voltage feature vectors.
3. Data split: stratified 70/15/15 with controlled seeds.
4. Training regimes: clean-only and noise-augmented.

### 7.5 Evaluation framework
1. Metrics: accuracy, macro-F1, per-class precision/recall, confusion matrix.
2. Robustness sweeps across severity multipliers (0.5x to 3x).
3. Statistical testing for key model comparisons.
4. Visual analysis via robustness curves and ablation heatmaps.

## 8. Work Plan and Timeline

### Phase A: Foundation (Weeks 1-3)
1. Verify EIDORS end-to-end execution and generate pilot samples.
2. Fix high-priority code issues that affect reproducibility and experiment validity.
3. Finalise corruption parameter set and class generation settings.

### Phase B: Dataset and baseline experiments (Weeks 4-6)
1. Generate full dataset target (~25,000 samples).
2. Train and evaluate baseline models under clean/noisy conditions.
3. Implement and validate CNN training pipeline.

### Phase C: Robustness and ablation studies (Weeks 7-9)
1. Run full clean/noisy cross-condition matrix.
2. Execute single-component and multi-component ablation experiments.
3. Run severity extrapolation tests and statistical analysis.

### Phase D: Writing and packaging (Weeks 10-13)
1. Draft dissertation chapters with integrated tables/figures.
2. Tie findings back to literature and stated research questions.
3. Finalise reproducibility artefacts and documentation.

### Phase E: Review and submission (Weeks 14-16)
1. Supervisor feedback integration.
2. Final quality checks and formatting.
3. Submission and post-submission paper preparation.

## 9. Success Criteria
The project will be considered successful if it achieves:

1. A fully reproducible data generation and model training pipeline.
2. Statistically supported evidence of robustness gains from noise-aware training.
3. Clear ablation insights into corruption sensitivity.
4. A coherent dissertation that meets all marking criteria, including critical analysis and ethical/sustainability discussion.
5. A publication-ready package (figures, methods, and code).

## 10. Risks and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| EIDORS setup or solver failures | High | Prioritise pilot run immediately; simplify to validated 2D if needed |
| Unrealistic corruption parameters | High | Ground parameters in literature and document assumptions explicitly |
| Overfitting to synthetic artefacts | High | Use cross-condition testing, severity sweeps, and ablation |
| Scope creep (3D, temporal models) | Medium | Treat as stretch goals only after core milestones are complete |
| Timeline compression | Medium | Use staged deliverables and fixed weekly checkpoints |

## 11. Ethical, Economic, and Sustainability Considerations
1. **Ethical**: Robust tactile inference in prosthetics can affect safety and user trust; limitations must be communicated clearly to avoid overclaiming clinical readiness.
2. **Economic**: A simulation-first framework can reduce experimental cost and accelerate early-stage model development.
3. **Sustainability**: Better modelling may reduce redundant hardware prototyping cycles; however, long-term sensor material durability and e-waste remain relevant concerns.

## 12. Deliverables
1. Dissertation manuscript (10,000 words, excluding references/appendices).
2. Reproducible repository with MATLAB/Python code and experiment scripts.
3. Results package: tables, confusion matrices, robustness curves, ablation plots.
4. Method documentation for corruption model and experimental protocol.
5. Draft journal manuscript outline based on dissertation findings.

## 13. Publication Pathway
Recommended progression:
1. Use dissertation outputs to prepare a manuscript focused on corruption-aware robustness in EIT touch classification.
2. Target a practical first venue such as *Sensors*.
3. Position an extended version for *IEEE Sensors Journal* if additional validation is obtained.

Proposed framing for publication:
> A reproducible corruption-aware training and evaluation framework for robust EIT tactile classification in prosthetic e-skin.

## 14. Expected Impact
If successful, this project will provide a robust and reproducible foundation for moving EIT tactile ML from idealised simulation settings towards more realistic deployment conditions. It will offer both immediate dissertation value and a credible springboard for publishable research.
