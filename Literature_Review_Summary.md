# Critical Summary of Literature Review Notes

## 1. Research Direction

The notes across approximately 72 sources converge on a clear dissertation topic: **machine learning for tactile perception in prosthetic electronic skin, with a particular emphasis on Electrical Impedance Tomography (EIT) as the sensing modality**. The likely research question is:

> *How can machine learning—particularly physics-informed and hybrid approaches—improve the robustness and accuracy of EIT-based tactile reconstruction for prosthetic skin, especially under realistic noise and deployment conditions?*

The notes consistently return to three pillars:
1. **Materials and sensor design** — hydrogels, conductive elastomers, self-healing substrates, and multi-material architectures that underpin e-skin hardware.
2. **EIT reconstruction** — the ill-posed inverse problem, classical regularisation (Tikhonov, FISTA, NOSER), and increasingly deep-learning-driven solutions (CNNs, U-Nets, autoencoders, attention mechanisms).
3. **Sim-to-real transfer and robustness** — domain randomisation, adversarial training, noise modelling, and sensor drift as barriers to real-world prosthetic deployment.

The notes suggest the student should emphasise:
- The gap between idealised simulation-trained models and real-world performance.
- The argument that hybrid physics+ML approaches (e.g. CNN-enhanced Tikhonov regularisation) are more suitable for safety-critical prosthetic applications than purely data-driven methods.
- Practical deployment constraints: power, latency, sensor degradation, and inter-user variability.

---

## 2. Gaps in the Literature

### 2.1 Identified through the notes

| Gap | Evidence from notes |
|-----|-------------------|
| **Sim-to-real transfer for EIT** | Nearly all ML-based EIT reconstruction studies train on EIDORS-generated simulation data. Papers [43], [44], [46] discuss domain randomisation and adversarial adaptation, but none apply these to EIT e-skin specifically. |
| **Realistic noise modelling** | Paper [33] shows that models trained with Gaussian noise alone are 5× more robust under noise but 100× worse on clean data. Paper [38] demonstrates that even 2% electrode drift causes visible distortion. Most papers use only additive white Gaussian noise — not multiplicative, drift, or spike noise. |
| **Long-term sensor degradation** | Papers [1], [18], [19] acknowledge material fatigue, conductivity changes, and hydrogel moisture loss, but no paper proposes learning-based compensation for these effects. |
| **Inter-user variability** | Papers [2], [56], [62] highlight that physiological differences across users drastically reduce generalisation, yet most datasets come from single-user or single-phantom setups. |
| **Multi-touch reconstruction** | Paper [20] addresses multi-touch EIT but with limited success. Deep learning approaches for simultaneous multi-contact localisation remain underdeveloped. |
| **Temporal modelling** | Papers [11], [57], [60] argue that temporal information is critical for dynamic tactile interaction, yet most EIT studies use static or quasi-static contact only. |
| **Co-design of materials and ML** | Paper [19] is one of very few works that jointly optimises material design and ML reconstruction. Most papers address only one axis. |
| **Standardised benchmarks** | No EIT-specific corruption benchmark exists analogous to ImageNet-C [47]. There is no standard dataset for comparing ML-based EIT reconstruction methods in tactile sensing. |

### 2.2 Broader literature gaps
- **Thermal regulation in prosthetic skin**: Paper [7] reviews biomimetic thermal materials but notes almost no adoption in prosthetics, despite clear effects on sensor accuracy.
- **Affective and social touch**: Papers [52]–[62] explore social touch recognition, but the connection between these high-level perception tasks and low-level EIT sensing is never made explicit.
- **Ethics and user experience**: No paper in the notes discusses ethical considerations, user consent, or the psychological impact of restored tactile feedback — a requirement of the marking scheme.

---

## 3. Gaps in the Review

### 3.1 Structural issues
- **Duplicate entries**: Papers [3] and [5] are identical; papers [44] and [45] are identical; papers [52] and [53] are identical. These should be removed.
- **No thematic grouping**: The notes are organised paper-by-paper (numbered 1–72) with no cross-referencing or thematic structure. The final dissertation review must be organised by theme, not by source.
- **No synthesis across sources**: Each paper is treated in isolation. There is no discussion of where sources agree, where they conflict, or how findings build upon one another.

### 3.2 Analytical weaknesses
- **Insufficient critical comparison of ML methods**: Many papers report high accuracies, but the notes rarely compare these across consistent conditions. For example, papers report ~91%–100% classification accuracies, but under incompatible setups (different tasks, datasets, noise conditions, validation strategies). This must be synthesised critically.
- **Limited discussion of evaluation methodology**: The notes identify missing cross-validation in individual papers but do not discuss this as a systemic problem across the field.
- **Shallow coverage of domain adaptation**: Papers [43], [44], [46] are included but their connection to the EIT problem is stated, not analysed.
- **Ethical, sustainable, and economic considerations**: The marking scheme explicitly requires these. The notes contain no such discussion.
- **Missing quantitative comparison table**: No structured comparison of ML architectures, datasets, metrics, or accuracies across the reviewed papers.

### 3.3 Missing source categories
- Papers on **spiking neural networks (SNNs)** for neuromorphic tactile processing — mentioned briefly in [6] but not explored.
- **Transfer learning** applied to tactile or EIT domains.
- **Transformer architectures** for tactile sensing — mentioned as a suggestion but no papers reviewed.
- **User studies or clinical trials** of prosthetic e-skin systems.

---

## 4. Strengths

The review notes demonstrate several notable strengths:

1. **Breadth of coverage**: 72 sources spanning materials science, sensor design, ML methodology, EIT theory, social touch, robustness, and domain adaptation. This reflects a wide and thorough literature search.

2. **Strong critical engagement at the individual paper level**: The student consistently identifies methodological weaknesses (missing cross-validation, small datasets, lack of noise robustness testing) — this is precisely the kind of critical analysis the mark scheme rewards.

3. **Clear identification of recurring limitations**: The notes repeatedly flag:
   - Simulation-only training data
   - Lack of generalisation evidence
   - Overly optimistic reported accuracies
   - Missing temporal modelling

4. **Good use of self-directed questions**: Prompts such as "Can joint-level strain signals be used to infer intended motion robustly enough to control a prosthetic, without EMG?" show intellectual curiosity and original thinking.

5. **Inclusion of foundational and methodological papers**: Papers on t-SNE [67], ROC analysis [68], evaluation metrics [69], ResNet [66], and empirical algorithm comparison [70] show methodological awareness beyond the immediate topic.

6. **Relevant technical depth in EIT**: Papers [15]–[20], [21]–[29], [36], [38]–[42] provide deep coverage of the EIT inverse problem, EIDORS tooling, and reconstruction algorithms — this gives the dissertation a strong technical foundation.

---

## 5. Weaknesses

| Weakness | Impact |
|----------|--------|
| **Paper-by-paper structure without synthesis** | Prevents the reader from seeing the broader argument; risks reading as an annotated bibliography rather than a literature review. |
| **Three duplicate entries** | Suggests incomplete organisation; will reduce word count efficiency in a 10,000-word dissertation. |
| **No explicit research questions or hypotheses drawn from the literature** | The review does not culminate in a clear gap statement that the dissertation will address. |
| **Inconsistent depth across papers** | Some entries have 12+ sections of detailed analysis; others have 2–3 lines. Peripheral papers (e.g. [7] on thermal materials) receive similar space to core papers (e.g. [19] on EIT e-skin). |
| **Limited quantitative synthesis** | No comparison table of ML methods, accuracies, datasets, or noise conditions across papers. |
| **No discussion of ethical, sustainable, or economic implications** | Required by the mark scheme (Critical Analysis criterion, 35 marks) and dissertation guidelines. |
| **Social touch papers [52]–[62] are loosely connected** | Their relevance to EIT-based prosthetic skin is not made explicit. If the dissertation does not address social touch, these may be unnecessary. |
| **No clear narrative arc** | The review does not tell a story from problem → existing approaches → limitations → gap → proposed contribution. |

---

## 6. Alignment with the Mark Scheme

| Criterion (Max marks) | Current alignment | Assessment |
|----------------------|-------------------|------------|
| **Scope** (10) | Aims are implicit but not stated. The review covers broad territory without clearly bounding the dissertation topic. | Likely **55–65%** — scope is evident but not sharply defined. |
| **Understanding of subject matter** (20) | Strong individual-paper understanding. Critical awareness of methodological weaknesses is a real strength. | Likely **65–75%** — good understanding, but synthesis across sources is lacking. |
| **Planning and implementation of methodology** (20) | The notes imply a simulation-based methodology (EIDORS, synthetic data, CNN reconstruction), but this is not explicitly articulated in the review. | Cannot fully assess from notes alone, but the review should connect methodology choices to the literature more directly. |
| **Critical analysis based on evidence** (35) | Strong at the individual source level. Weak on cross-source synthesis, contradiction identification, and original argument construction. Missing ethical/sustainable discussion. | Likely **55–65%** — analytical potential is clear, but execution needs restructuring. |
| **Presentation / communication** (10) | Notes are not in a presentable state (paper-by-paper, inconsistent formatting, duplicates). | Not applicable to notes, but the final review must be Harvard-referenced, logically structured, and visually clear. |
| **Spelling, grammar, syntax** (5) | Generally clear writing in the notes. Minor inconsistencies. | Likely adequate if carried through to the final document. |

**Overall projected band based on notes alone: Merit (60–69%)**, with potential to reach Distinction (70%+) if the synthesis, structure, and critical argument are significantly strengthened.

---

## 7. Impact of Previous Feedback

No formal prior feedback is referenced within the notes. However, the mark scheme and dissertation guidelines provide implicit feedback on expectations. The following improvements should be drawn from these:

| Guideline/Criterion requirement | Practical improvement |
|--------------------------------|----------------------|
| "Has the student been sufficiently critical with respect to other people's arguments and published 'facts'?" | Move from noting limitations to **arguing why they matter** for the dissertation's specific contribution. |
| "Are the results and analysis tied back to the literature?" | Build a comparison framework now so that results can be directly compared to the papers reviewed. |
| "Consideration of ethical, sustainable, or economic implications" | Add a subsection on: (a) the ethics of restoring tactile sensation, (b) the environmental cost of e-skin materials, (c) economic accessibility of prosthetic sensing systems. |
| "Does the discussion evaluate and critique the research outcomes and propose any new hypotheses?" | Use the literature gaps to derive 2–3 testable hypotheses that the dissertation will address. |
| Milestone: "Submit the draft of your Introduction chapter by 3rd July" | The review must be sufficiently complete to support the introduction. Prioritise structuring the argument now. |

---

## 8. Recommended Next Steps

1. **Remove duplicates** (papers [3]/[5], [44]/[45], [52]/[53]) and decide which social touch papers are essential.

2. **Restructure the review thematically**. Suggested structure:
   - Section 2.1: Electronic skin for prosthetics — materials, requirements, and biological inspiration
   - Section 2.2: EIT as a sensing modality — principles, advantages, and inherent limitations
   - Section 2.3: ML for EIT reconstruction — classical, deep learning, and hybrid approaches
   - Section 2.4: Robustness, noise, and the sim-to-real gap
   - Section 2.5: Touch classification and perception — from static sensing to temporal and social touch
   - Section 2.6: Deployment challenges — scalability, drift, inter-user variability, ethical considerations

3. **Create a comparison table** summarising key papers by: sensing modality, ML method, dataset type (real/simulated), validation strategy, reported accuracy, and identified limitations.

4. **Write explicit gap statements** at the end of each thematic section, culminating in a summary of the overall research gap the dissertation addresses.

5. **Add ethical, sustainable, and economic discussion** — even a brief paragraph will satisfy the mark scheme requirement and demonstrate maturity.

6. **Derive 2–3 specific research questions or hypotheses** from the gaps identified. For example:
   - *RQ1: Does physics-informed noise augmentation (electrode drift, conductivity variation) during training improve EIT reconstruction robustness compared to Gaussian noise alone?*
   - *RQ2: Can a hybrid CNN-regularisation approach outperform end-to-end deep learning for multi-touch EIT reconstruction under realistic noise conditions?*

7. **Strengthen cross-source synthesis**: For every claim, cite at least two sources and explain whether they agree, disagree, or extend one another.

8. **Ensure Harvard referencing** is applied consistently. The notes currently use informal numbering ([1], [2], etc.) which must be replaced.

---

## Thematic Summary Table

| Theme | What the notes show | Implication for the dissertation |
|-------|---------------------|----------------------------------|
| **EIT fundamentals** | Well-covered (papers [15]–[20], [36], [38]–[42]). Strong grasp of forward/inverse problem, EIDORS, regularisation. | Provides robust technical foundation. Ensure this is synthesised, not merely listed. |
| **ML for EIT reconstruction** | Extensive coverage (papers [21]–[35]). Ranges from early ANNs to attention-based CNNs. | Must compare architectures on consistent criteria. Argue *why* a hybrid approach is appropriate. |
| **Materials and sensor design** | Good breadth ([1], [3]/[5], [4], [9], [19], [49]). Self-healing, variable sensitivity, and multi-material skins covered. | Link material properties to ML implications (e.g. how self-healing affects model retraining needs). |
| **Robustness and domain transfer** | Conceptually strong ([43]–[48]), but not applied to EIT. | This is the most promising area for an original contribution. Make this connection explicit. |
| **Social and affective touch** | Extensive ([52]–[62]) but loosely connected to the core topic. | Either integrate meaningfully (e.g. as a downstream application of EIT sensing) or reduce coverage. |
| **Deployment and practical constraints** | Repeatedly mentioned (power, latency, scalability) but not systematically analysed. | Dedicate a focused subsection to deployment constraints, linking each to specific ML design choices. |
| **Ethics and sustainability** | Absent. | Must be added to satisfy the marking criteria. |

---

## Overall Verdict

The literature review notes demonstrate **considerable breadth and genuine critical awareness** at the individual-source level. The student has read widely, identified recurring methodological weaknesses in the field, and shown intellectual curiosity about open problems. The technical depth in EIT and ML reconstruction is a particular strength.

However, in its current form, the review reads as an **annotated bibliography rather than a critical literature review**. The critical step from *noting limitations in individual papers* to *constructing a coherent argument about the state of the field* has not yet been taken. The notes lack thematic structure, cross-source synthesis, explicit gap statements, and any discussion of ethical or sustainable considerations.

With focused restructuring, synthesis, and the addition of a clear narrative arc from the identified gaps to the dissertation's contribution, this review has strong potential to reach the **lower Distinction band (70–74%)**. Without these changes, it is likely to remain in the **Merit range (60–69%)**.
