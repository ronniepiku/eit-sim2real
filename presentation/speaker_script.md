# EE52037 — presentation script
**Ronald Piku · A Physically Motivated Noise Model for Robust EIT Tactile Classification Under Simulated Domain Shift**

Total 951 words ≈ **7:03 at 135 words/min** (range: 6.6–7.6 min across 125–145 wpm). Slide 11 (References) is not narrated — let it sit on screen during the closing line.

Markers show elapsed time at the *start* of each slide, at 135 wpm.

---

## Slide 1 — Title – 0:00 – ~12s

Hi, I'm Ronald Piku. Thank you for joining me as I present my dissertation: a physically motivated noise model for robust EIT tactile classification under simulated domain shift.

---

## Slide 2 — Introduction – 0:12 – ~23s

Here's how this presentation is structured. First, I define the problem this
creates and what I mean by the simulation-to-reality gap. Then three results
follow: what closes that gap, how it was evaluated, and the smallest noise
model that still works. Finally, I review the conclusions and discuss the
work's limitations.

---

## Slide 3 — The problem – 0:35 – ~56s

Around fifty-eight million people live with limb loss, and more than a third
abandon a powered prosthetic arm — the most cited reason is that the device
itself has no sense of touch, so it can't tell when or how it is being
handled.

Electrical Impedance Tomography, or EIT, gives it that awareness. As Figure 1
shows, sixteen electrodes sit around a soft sensing disc covering the arm,
modelled on an ionic hydrogel skin (Costa Cornellà et al., 2023) for its
sensing fidelity, even though it isn't the most durable option available.
Inject a small current between one pair, measure the voltage at another —
where somebody presses, that voltage shifts, and the full set of
measurements is a vector you can feed straight into a classifier.

---

## Slide 4 — Where the field is – 1:32 – ~47s

Published EIT touch classifiers report ninety-eight, ninety-nine per cent
accuracy, almost all trained on simulated measurements, because labelled real
data is expensive to collect.

But a simulator gives you a perfect measurement, and hardware never does —
the same gap Tobin and colleagues closed in robotics and computer vision by
training on deliberately varied simulation, and the gap this research aims to
close for EIT. Within EIT, four sources dominate: thermal noise, contact
impedance, electrode placement, and quantisation.

When prior work adds noise at all, it's almost always just Gaussian noise.
This project asks whether that's enough, and if not, which of the four
actually matter.

---

## Slide 5 — The gap, made visible – 2:19 – ~35s

Before any classification, here's the problem in one picture. On the left,
Figure 2 shows five touch types reconstructed from clean simulated
measurements: you can see exactly where the contact is in each one, and each
class is clearly separable by its pressure intensity and contact area.

On the right, the same five with realistic noise switched on. The clear
distinction between classes has gone — real-world classification is
difficult even to the human eye, let alone a classifier.

---

## Slide 6 — Result 1 – 2:54 – ~47s

To see how serious this gap is, I trained four classifiers — a random
forest, a support vector machine, a multilayer perceptron and a 1D
convolutional neural network — each on 25,000 simulated touches, clean and
noisy in three combinations.

As Figure 3 shows, on clean data the network reaches ninety-four per cent,
close to published results. Trained clean but tested noisy, it collapses to
twenty per cent — as good as guessing. Train and test on noisy data instead,
and it recovers to seventy-six: a fifty-five-point gain. That pattern held
for all four classifiers, so it isn't something a bigger model fixes on its
own.

---

## Slide 7 — Result 2, the main contribution – 3:41 – ~51s

The ablation study tests each noise source alone. There are two ways to
score it.

Test each model on the same noise it trained with — the blue bars in Figure
4 — and Gaussian noise looks harmless at ninety-seven per cent, electrode
bias the most damaging.

Or test against the full corruption a real device produces — the orange
bars. The story reverses: electrode bias is the only source anywhere close
to surviving, the rest fall close to chance. That's not a surprise from
nowhere — electrode positioning error is already flagged in the EIT
literature as a dominant modelling error (Kolehmainen et al., 1997). This
result confirms it, under the correct protocol.

---

## Slide 8 — Result 3 – 4:32 – ~57s

Using the correct protocol, two of the four components carry everything:
electrode bias plus Gaussian noise reaches 76.5 per cent, level with the full
model — the noise model halves with no loss of accuracy.

That decomposes further. The 'no contact' class in simulation is an exact
vector of zeros, so I removed it and reran on the four real touches alone.
Electrode bias alone then matches the full model — Gaussian noise was never
providing robustness, just making a class that's zero in training
recognisable when it arrives noisy.

Figure 5 shows why that matters: no-contact is recovered perfectly, a direct
consequence of that Gaussian floor, while point contact is the weakest
class. One component to discriminate between touches, a noise floor to
detect touch at all.

---

## Slide 9 — Limitations – 5:28 – ~54s

To be upfront about the boundaries: I can claim which noise sources matter
and why, that the conventional protocol inverts that answer, a mechanism for
the architecture result that's tested rather than assumed, and a pipeline
anyone can rerun.

I cannot claim any of it survives contact with hardware. Every number is
simulated in 2D — a real device isn't, and I only classify single, static
frames, so drift over time isn't represented at all. My electrode bias is
drawn independently per electrode, and on a stretched sheet it would be
spatially correlated — measuring that covariance is the first experiment I'd
run next. And seventy-six per cent isn't deployable: detecting contact is
reliable, saying which touch it was is not.

---

## Slide 10 — Close – 6:22 – ~31s

To pull that together: realistic noise collapses accuracy from ninety-four
to twenty per cent, and training on it recovers seventy-six. How you
evaluate an ablation decides its conclusion, not just its result — the
finding I'd defend hardest.

In one sentence: noise-aware training closes most of the gap, but only if
you test it against the world the device will actually meet. Thank you, I'm
happy to take questions.

---

## Slide 11 — References – 6:53 – ~10s, not narrated

Full references for every claim in this talk, in Harvard style, are listed
here for anyone who would like to follow up.

---

## Recording notes

- **Slides 7 and 8 carry the marks.** Critical Thinking and Argumentation is
  70% of this assessment. Slide 7 is the protocol-inversion finding (the
  main contribution) and slide 8 decomposes it further with the confusion
  matrix. Do not rush these to save time elsewhere.
- **Rehearsal note on research reflexivity.** The self-testing content that
  used to be its own slide (the permutation test and the ρ-sweep,
  confirming the CNN's advantage is really about measurement structure, and
  that the electrode-bias result isn't inflated by cancellation) is no
  longer in the pre-recorded deck. Both checks are still in the
  dissertation (Section on testing assumptions) and are strong, concrete
  material for the 13-minute viva if a question about evidence or
  robustness comes up — keep them ready to describe verbally.
- **Numbers to land clearly:** 94 → 20 → 76 (slide 6); the matched vs
  deployment inversion (slide 7); the no-contact recovery in the confusion
  matrix (slide 8).
- **Slide 11 (References)** needs no delivery time — say the thank-you/
  questions line while it's on screen.
