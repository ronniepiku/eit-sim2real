# EE52037 — 7-minute presentation script
**Ronald Piku · A Physically Motivated Noise Model for Robust EIT Tactile Classification Under Simulated Domain Shift**

Target ~6:50 at a normal speaking pace (~135 words/min). Total 923 words,
leaving roughly ten seconds of headroom against the 7-minute limit.
Timings assume you pause briefly on each slide change — those pauses are already
in the budget. If you naturally speak fast, slow down rather than adding words.

Markers show elapsed time at the *start* of each slide.

---

## Slide 1 — Title · 0:00 · ~8s

I'm Ronald Piku. This project asks why EIT touch classifiers that work in
simulation fail on real hardware.

---

## Slide 2 — The problem · 0:08 · ~45s

Around fifty-eight million people live with limb loss, and more than a third
abandon a powered prosthetic hand. The most cited reason is that it gives
nothing back. You cannot feel whether you're holding an egg or crushing it.

The sensing approach here is Electrical Impedance Tomography. Take a sheet of
soft conductive material, put sixteen electrodes round the edge, pass a small
current across it and measure the voltages. Where somebody presses, the
resistance changes and the voltages shift. One sheet, no wiring through the
sensing surface, which is what makes it attractive for a prosthetic.

---

## Slide 3 — Where the field is · 0:53 · ~48s

Published EIT touch classifiers report ninety-eight or ninety-nine per cent
accuracy, almost all trained on simulated measurements, because labelled real
data is slow and expensive.

But a simulator gives you a perfect measurement and hardware does not. Four
things get added. Thermal noise in the amplifiers. Contact impedance, which
differs electrode to electrode. Errors in where the electrodes physically sit.
And quantisation, from the converter.

When prior work adds noise at all, it adds Gaussian noise — the easy one. My
question was whether that's enough, and if not, which of the four actually
matter.

---

## Slide 4 — The gap, made visible · 1:41 · ~42s

Here's the problem before any machine learning. On the left, five touches
reconstructed from clean simulated measurements. You can see where the finger is
in each one.

On the right, the same five with the noise switched on. The contact has gone.
And what replaced it isn't random speckle — it has structure, those quadrant
seams, because the dominant error attaches to individual electrodes, and each
electrode feeds a whole block of measurements. That structure matters, and I'll
come back to it.

---

## Slide 5 — Result 1 · 2:23 · ~38s

I generated twenty-five thousand simulated touches and trained four classifiers
on clean data. On clean test data the network reaches ninety-four per cent — the published
result reproduces. Show it realistic measurements and it falls to twenty, which
with five classes is exactly guessing. It hasn't degraded; it has stopped
working.

That held for all four classifiers and all three feature representations, so
it's not fixed by a bigger model. Train on realistic noise instead and you
recover seventy-six per cent.

---

## Slide 6 — Result 2, the main contribution · 3:01 · ~65s

Now the main contribution, which began as my own mistake.

To find which noise sources matter, you remove them one at a time. The question
is what you then test against. The intuitive choice — the one I made first, and
the one the literature uses — is to test each model on the same noise it was
trained with. That's the blue bars. Read them and you'd conclude Gaussian noise
is harmless at ninety-seven per cent, and electrode placement is the worst thing
on the list.

The orange bars test those same models against the full corruption a real device
produces. The ranking inverts. Gaussian-only drops to chance. The electrode-bias
model is the only one that survives.

Blue is measuring the difficulty of an artificial world containing one problem.
It rewards a model for defining an easier test for itself. Had I reported only
that column, every piece of design guidance in my dissertation would have been
backwards.

---

## Slide 7 — Result 3 · 4:06 · ~52s

Using the correct protocol, two of the four components carry everything.
Electrode bias plus Gaussian noise reaches seventy-six and a half — level with
the full four-component model. So the noise model halves.

But I was suspicious of *why* Gaussian noise was needed, because the no-contact
class in simulation is an exact vector of zeros. It's degenerate. So I removed
that class and re-ran on the four real touches. Electrode bias alone then
matches the full model, and adding Gaussian changes nothing.

So Gaussian noise was never providing robustness. It was making a class that's
perfectly zero in training recognisable when it arrives noisy. One component to
discriminate between touches, a noise floor to detect touch at all. That's
sharper than my original claim, and less flattering.

---

## Slide 8 — Testing my own work · 4:58 · ~57s

Two things in that argument were assumptions rather than evidence, so I tested
them.

First, I claimed the network wins by exploiting the block structure. That was an
inference about my own model. So I shuffled the measurement order — identical
information, adjacency destroyed. The network loses thirty points. The other
three don't move, and the support vector machine is identical to the decimal
place, which confirms no information was lost. Demonstrated, not asserted.

Second, and less comfortably: two of the papers I cite show this kind of
electrode error largely cancels when you subtract a reference frame, which is
what my pipeline does. If that held here, my central result would be inflated.
So I made the cancelling fraction a parameter and swept it. It holds until
cancellation is essentially perfect — which is the condition those same papers
say fails in practice.

---

## Slide 9 — Limitations · 5:55 · ~40s

On the boundaries. I can claim which noise sources matter and why, that the
conventional protocol inverts that answer, a tested mechanism for the
architecture result, and a pipeline anyone can rerun.

I cannot claim any of it survives hardware. Every number is simulated. My
electrode bias is independent per electrode; on a stretched sheet it would be
correlated, and measuring that covariance is the first experiment I'd run. And
seventy-six per cent isn't deployable — detecting that contact happened is
reliable, saying which touch it was is not.

---

## Slide 10 — Close · 6:35 · ~15s

Noise-aware training closes most of the gap, but only if you test against the
world the device will actually meet. The conventional alternative gives you the
opposite answer, confidently.

Thank you.

---

## Recording notes

- **Pace.** The budget assumes ~135 wpm. Record slide 6 first as a calibration
  test: if it runs longer than 70 seconds you are over budget everywhere.
- **Slide 6 and slide 8 carry the marks.** Critical Thinking and Argumentation is
  70% of this assessment. Those two slides are where you demonstrate ownership of
  the work and willingness to test yourself. Do not rush them to save time
  elsewhere — cut slide 2 or 3 instead.
- **Say "my own mistake" out loud on slide 6.** It is not a weakness to admit;
  the marking sheet explicitly rewards research reflexivity, and a correction you
  found yourself is stronger evidence of understanding than a result that went
  right first time.
- **Numbers to land clearly:** 94 → 20 → 76 (slide 5); the inversion (slide 6);
  30-point drop (slide 8). Everything else can be approximate in delivery.
- **If you overrun**, cut in this order: the "no wiring" clause on slide 2, the
  four-component list on slide 3 (the slide already shows it), then the last
  sentence of slide 5. Never cut from slides 6, 7 or 8.
