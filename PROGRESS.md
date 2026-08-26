# Dissertation Progress

Last updated: 2026-08-26

## Word count

CLAUDE.md rule: 10,000 words ±1,000 (9,000 floor / 11,000 hard ceiling), counting
abstract, footnotes and all chapters; excluding figures, table captions,
references and title pages.

Title page currently states **11,829** words (Abstract to Conclusions). An
independent prose-only estimate this session, stripping LaTeX markup and all
figure/table/equation environments, gives **~11,426**:

| Section | Est. words |
|---|---:|
| Abstract | 265 |
| Introduction | 1,147 |
| Literature Review | 1,900 |
| Methodology | 2,678 |
| Results | 2,548 |
| Discussion | 1,985 |
| Conclusions | 903 |
| **Total** | **~11,426** |

Both figures sit above the 11,000 hard ceiling. Neither number is authoritative
(no `texcount`/`detex` available in this environment; the estimate is a
regex-based approximation) — see Outstanding Issues #1.

## Chapter status

All six chapters and both appendix chapters are complete, internally
consistent, and well-integrated with the literature review's stated gaps. No
`[TODO]`, placeholder, or draft comment found anywhere in `dissertation/`.

| Chapter | Status | Notes |
|---|---|---|
| Abstract | Done | Matches conclusions' claims |
| Introduction | Done | Aims/objectives map cleanly to Results/Conclusions |
| Literature Review | Done | Table 2.2/2.3/2.4 (gaps summary) tie directly to contributions |
| Methodology | Done | Longest chapter; candidate for trimming if word count must come down |
| Results | Done | Longest chapter; candidate for trimming if word count must come down |
| Discussion | Done | Limitations section already names "no hardware validation" as top priority future work |
| Conclusions | Done | Objectives explicitly revisited; reflections section is candid about a caught pipeline bug |
| Appendices (A–D) | Done | Cross-validation corroboration, Gaussian-only ablation, EDA, baseline configs, repo link |

## Outstanding issues (priority order)

1. **Word count likely exceeds the 11,000 hard ceiling.** Both the stated
   (11,829) and independently estimated (~11,426) figures are over. Needs a
   proper word count (e.g. via Overleaf's counter or `texcount` if available)
   to confirm the true number, then trimming — Methodology and Results are
   the longest chapters and the likely trim targets. Not yet actioned; needs
   your go-ahead before any cutting, per CLAUDE.md.

2. **Pre-submission checklist not yet run this session.** The 13-item list in
   CLAUDE.md (citation/reference consistency, figure/table numbering,
   notation consistency, acronym consistency, abstract-vs-conclusion
   alignment, etc.) hasn't been worked through yet.

3. **Presentation deck out of sync with the new title.** `build_deck.py` was
   edited today to match the new dissertation title but hasn't been re-run,
   so the built `.pptx`/slide images still show the old title until
   regenerated.

## Recent changes

- 2026-08-26: Title changed from *"Towards Simulation-to-Reality Transfer in
  EIT Tactile Sensing: A Noise-Augmented Deep Learning Approach"* to *"A
  Physically Motivated Noise Model for Robust EIT Tactile Classification
  Under Simulated Domain Shift"*, on the grounds that the dissertation's own
  Limitations section states no hardware validation was performed and results
  "remain matched-distribution figures rather than demonstrated sim-to-real
  performance." Updated consistently in `dissertation/main.tex` (title +
  title page), `README.md` (heading + BibTeX citation),
  `presentation/speaker_script.md`, `presentation/build_deck.py`, and
  `matlab/main.m` header comment.

## Not yet reviewed in depth this session

- Citation/reference cross-check (every in-text citation in `references.bib`
  and vice versa).
- Figure/table numbering and caption completeness (visual check — figures
  weren't rendered, only their LaTeX source and captions read).
- Notation/units/acronym consistency across chapters.
- Presentation content vs. dissertation vs. presentation mark scheme.
