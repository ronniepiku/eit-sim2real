"""Build the EE52037 7-minute viva presentation.

Deck is deliberately sparse: one claim per slide, a single visual carrying the
evidence, and the argument delivered in the voiceover rather than read off the
screen. Speaker notes carry the timed script.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = Path(__file__).parent
FIG = HERE / "figures"
REC = HERE.parent / "results" / "dataset_validation" / "reconstructions"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
RULE = RGBColor(0xE6, 0xE6, 0xE3)

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = SURFACE
    return s


def tb(s, x, y, w, h, text, size=18, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space_after=6, line=1.15):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = "Calibri"
    return box


def heading(s, title, kicker=None):
    if kicker:
        tb(s, 0.85, 0.52, 11.6, 0.4, kicker.upper(), size=13, bold=True, color=BLUE)
        tb(s, 0.85, 0.92, 11.6, 0.9, title, size=32, bold=True)
    else:
        tb(s, 0.85, 0.72, 11.6, 0.9, title, size=32, bold=True)


def rule(s, y=1.92):
    ln = s.shapes.add_shape(1, Inches(0.85), Inches(y), Inches(11.6), Pt(1.6))
    ln.fill.solid()
    ln.fill.fore_color.rgb = RULE
    ln.line.fill.background()
    ln.shadow.inherit = False


def pic(s, path, x, y, w, crop_top=0.0):
    p = s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    if crop_top:
        p.crop_top = crop_top
    return p


def note(s, text):
    s.notes_slide.notes_text_frame.text = text


def num(s, n):
    tb(s, 12.35, 6.92, 0.6, 0.3, str(n), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════ 1 — title
s = slide()
tb(s, 0.85, 2.15, 11.6, 0.4, "MSc Artificial Intelligence in Engineering and Design",
   size=15, bold=True, color=BLUE)
tb(s, 0.85, 2.75, 11.2, 1.9,
   "Towards Simulation-to-Reality Transfer\nin EIT Tactile Sensing",
   size=40, bold=True, line=1.1)
tb(s, 0.85, 4.55, 11.2, 0.5, "A noise-augmented deep learning approach",
   size=21, color=MUTED)
tb(s, 0.85, 5.65, 11.2, 1.0,
   "Ronald Piku       Supervisor: Prof. Manuchehr Soleimani\n"
   "Department of Electronic and Electrical Engineering, University of Bath",
   size=14, color=MUTED, line=1.5)
note(s, """[0:00-0:08]

Hello. I'm Ronald Piku. This project asks a question about electronic skin for
prosthetic limbs: why do touch classifiers that work beautifully in simulation
fail on real hardware, and what does it take to fix that.""")

# ═══════════════════════════════════════════════════════ 2 — the problem
s = slide()
heading(s, "Prosthetic hands can grip. They cannot feel.", kicker="The problem")
rule(s)
tb(s, 0.85, 2.35, 5.3, 3.4,
   "57.7 million people live with limb loss.\n\n"
   "More than a third abandon myoelectric\nprostheses — the absence of touch\nis a leading reason.\n\n"
   "Without feedback, the user cannot tell\na firm grip from a crushing one\nwithout looking at it.",
   size=19, line=1.35, space_after=2)
tb(s, 6.9, 2.35, 5.6, 3.4,
   "Electrical Impedance Tomography\nturns a whole sheet of soft material\ninto one continuous touch sensor.\n\n"
   "16 electrodes around the rim.\nInject a small current, measure the\nvoltages, infer where the pressure is.\n\n"
   "No wires threaded through the skin.",
   size=19, color=MUTED, line=1.35, space_after=2)
num(s, 2)
note(s, """[0:08-0:53]  ~45s

Around fifty-eight million people worldwide live with limb loss. More than a
third of them abandon a powered prosthetic hand, and the single most cited
reason is that it gives them nothing back. You cannot feel whether you are
holding an egg or crushing it, so you have to watch your own hand do it.

The sensing approach I work on is Electrical Impedance Tomography, and the idea
is simple. Take a sheet of soft, slightly conductive material and put sixteen
electrodes around the edge. Pass a tiny current across it, measure the voltages
at the rim, and where somebody presses, the electrical resistance changes and
the voltages shift. One sheet, sixteen wires at the border, and no wiring
threaded through the sensing surface itself. That is what makes it attractive
for a prosthetic.""")

# ═══════════════════════════════════════════════════════ 3 — why simulate
s = slide()
heading(s, "Training data is simulated — and simulation is too clean",
        kicker="Where the field is")
rule(s)
tb(s, 0.85, 2.3, 11.6, 0.6,
   "Published EIT touch classifiers report 98–99% accuracy. Almost all are trained on "
   "simulated measurements, because labelled real data is expensive.",
   size=19, color=MUTED, line=1.3)
tb(s, 0.85, 3.4, 11.6, 0.45,
   "Real hardware adds four things the simulator does not:", size=19, bold=True)
for i, (t, d) in enumerate([
        ("Thermal noise", "in the amplifiers"),
        ("Contact impedance", "varies electrode to electrode"),
        ("Electrode placement", "is never exactly where the model assumes"),
        ("Quantisation", "from the analogue-to-digital converter")]):
    y = 4.05 + i * 0.62
    tb(s, 1.15, y, 3.3, 0.45, t, size=18, bold=True, color=BLUE)
    tb(s, 4.5, y, 8.0, 0.45, d, size=18, color=MUTED)
tb(s, 0.85, 6.62, 11.6, 0.5,
   "Prior work augments with one of them — Gaussian noise — if any.",
   size=19, bold=True)
num(s, 3)
note(s, """[0:53-1:43]  ~50s

Published EIT touch classifiers report accuracies around ninety-eight, ninety-nine
per cent. Almost all of them are trained on simulated measurements, because
collecting labelled real touch data is slow and expensive.

The trouble is that a simulator gives you a perfect measurement, and real
hardware does not. Four things get added on the way. Thermal noise in the
amplifiers. Contact impedance, which differs from electrode to electrode. Errors
in where the electrodes physically sit, which are never exactly where the model
assumes. And quantisation, from the analogue-to-digital converter.

When prior work adds noise at all, it adds one of these — Gaussian noise, the
easy one. My question was whether that is enough, and if not, which of the four
actually matter.""")

# ═══════════════════════════════════════════════════════ 4 — what noise does
s = slide()
heading(s, "This is what realistic noise does to the signal",
        kicker="The gap, made visible")
rule(s)
tb(s, 1.35, 2.15, 4.6, 0.4, "Clean simulation", size=18, bold=True, color=BLUE,
   align=PP_ALIGN.CENTER)
tb(s, 7.35, 2.15, 4.6, 0.4, "With realistic noise", size=18, bold=True, color=ORANGE,
   align=PP_ALIGN.CENTER)
pic(s, REC / "random_reconstructed_class_images_clean.png", 0.9, 2.65, 5.5, crop_top=0.06)
pic(s, REC / "random_reconstructed_class_images_noisy.png", 6.9, 2.65, 5.5, crop_top=0.06)
tb(s, 0.85, 6.5, 11.6, 0.5,
   "Same five touches. The contact is plainly visible on the left and gone on the right.",
   size=18, color=MUTED)
num(s, 4)
note(s, """[1:43-2:28]  ~45s

Before any machine learning, here is the problem in one picture. On the left are
five different touches reconstructed from clean simulated measurements — no
contact, a light touch, a firm press, a fingertip, and a broad contact. You can
see exactly where the finger is in each one.

On the right are the same five touches, with the four noise sources switched on.
The contact has disappeared. What you are looking at instead is the noise
itself — and notice it is not random speckle. It has visible structure, those
quadrant seams, because the dominant error attaches to individual electrodes and
each electrode contributes to a whole block of measurements.

That structure turns out to matter enormously, and I will come back to it.""")

# ═══════════════════════════════════════════════════════ 5 — the collapse
s = slide()
heading(s, "Every model I tried collapsed to guessing", kicker="Result 1")
rule(s)
pic(s, FIG / "p_collapse.png", 1.45, 2.15, 10.4)
tb(s, 0.85, 6.45, 11.6, 0.6,
   "Four architectures, three feature representations, same collapse. "
   "It is not a capacity problem — it is a distribution problem.",
   size=18, color=MUTED, line=1.3)
num(s, 5)
note(s, """[2:28-3:08]  ~40s

So I built the noise model, generated twenty-five thousand simulated touches,
and trained four classifiers on clean data.

On clean test data the network reaches ninety-four per cent — the published
result reproduces. Show it realistic measurements and it falls to twenty per
cent, which with five classes is exactly guessing. It has not degraded; it has
stopped working.

That happened for all four classifiers and all three ways of representing the
data, which tells you something useful: this is not fixed by a bigger model or
better features. The training distribution simply does not contain the
deployment distribution. Train on realistic noise instead and you recover to
seventy-six per cent.""")

# ═══════════════════════════════════════════════════════ 6 — protocol
s = slide()
heading(s, "How you test an ablation decides what it tells you", kicker="Result 2 — the main contribution")
rule(s)
pic(s, FIG / "p_inversion.png", 1.45, 2.05, 10.4)
tb(s, 0.85, 6.4, 11.6, 0.7,
   "Blue is the conventional protocol and it is misleading: it rewards a model for "
   "defining an easier world, not for surviving a real one.",
   size=18, color=MUTED, line=1.3)
num(s, 6)
note(s, """[3:08-4:13]  ~65s

Now the part I think is the real contribution, and it began as my own mistake.

To find out which noise sources matter, you remove them one at a time. The
question is what you then test against. The intuitive choice — the one I made
first, and the one the literature uses — is to test each model on the same
single noise source it was trained with. That is the blue bars. Read them and
you would conclude Gaussian noise is harmless, at ninety-seven per cent, and
electrode placement is the worst thing on the list at seventy-nine.

The orange bars test the same four models against the full corruption a real
device actually produces. The ranking inverts completely. The Gaussian-only
model drops to twenty-one per cent — chance. The electrode-bias model is the
only one that survives, at fifty-eight.

The blue protocol is measuring the difficulty of an artificial world containing
one problem. It rewards a model for having defined an easier test for itself.
Had I reported only that column, every piece of design guidance in my
dissertation would have been precisely backwards — and this applies to component
ablations generally, not just to EIT.""")

# ═══════════════════════════════════════════════════════ 7 — minimal model
s = slide()
heading(s, "Two components are enough — and they do different jobs", kicker="Result 3")
rule(s)
tb(s, 0.85, 2.3, 11.6, 0.55,
   "Electrode bias + Gaussian noise reaches 76.5% — level with the full four-component model. "
   "Contact impedance and quantisation add nothing measurable.",
   size=19, line=1.3)
box = s.shapes.add_shape(1, Inches(0.85), Inches(3.5), Inches(11.6), Inches(2.5))
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0xF2, 0xF4, 0xF7)
box.line.fill.background()
box.shadow.inherit = False
tb(s, 1.25, 3.78, 10.8, 0.45, "But that result decomposes further — and it qualified my own headline",
   size=18, bold=True, color=ORANGE)
tb(s, 1.25, 4.35, 10.8, 1.5,
   "The 'no contact' class is an exact zero vector in simulation. Remove it, and electrode bias "
   "alone matches the full model (70.5% vs 69.8%).\n\n"
   "Gaussian noise was never conferring robustness. It was making one degenerate class detectable.",
   size=18, line=1.35, space_after=8)
tb(s, 0.85, 6.35, 11.6, 0.6,
   "One component to tell touches apart. A noise floor to notice touch at all.",
   size=19, bold=True, color=BLUE)
num(s, 7)
note(s, """[4:13-5:08]  ~55s

Using the correct protocol, the answer is that two of the four components carry
everything. Electrode bias plus Gaussian noise reaches seventy-six and a half per
cent, statistically level with the full four-component model. Contact impedance
and quantisation contribute nothing I can measure. So the noise model can be
halved.

But I was suspicious of why Gaussian noise was needed at all, because the
'no contact' class in my simulation is an exact vector of zeros — it is
degenerate. So I removed that class and ran the ablation again on the four real
touches. Electrode bias on its own then matches the full model, and adding
Gaussian noise changes nothing.

So Gaussian noise was never providing robustness. It was doing one job: making a
class that is perfectly zero in training recognisable when it arrives noisy at
test time. The honest statement is one component to discriminate between touches,
and a noise floor to detect touch at all. That is a sharper claim than the one I
started with, and it is less flattering.""")

# ═══════════════════════════════════════════════════════ 8 — assumptions
s = slide()
heading(s, "Two assumptions I did not want to take on trust", kicker="Testing my own work")
rule(s)
tb(s, 0.85, 2.15, 5.5, 0.45, "Is it really the structure?", size=19, bold=True, color=INK)
pic(s, FIG / "p_permutation.png", 0.85, 2.65, 5.6)
tb(s, 6.95, 2.15, 5.5, 0.45, "Does differencing cancel it?", size=19, bold=True, color=INK)
pic(s, FIG / "p_rho.png", 6.95, 2.65, 5.6)
tb(s, 0.85, 5.75, 5.6, 1.1,
   "Shuffle the measurement order — same information, no adjacency. "
   "Only the CNN suffers. The mechanism is confirmed, not assumed.",
   size=16, color=MUTED, line=1.25)
tb(s, 6.95, 5.75, 5.6, 1.1,
   "Two of my own sources say static electrode error cancels. "
   "Sweeping that fraction: the result holds until cancellation is near-total.",
   size=16, color=MUTED, line=1.25)
num(s, 8)
note(s, """[5:08-6:03]  ~55s

Two things in that argument were assumptions rather than evidence, so I tested
them.

First, I claimed the network wins because it exploits the block structure of the
noise. That was an inference about my own model. So I shuffled the order of the
two hundred and eight measurements — identical information, adjacency destroyed —
and retrained. The network loses thirty points. The other three classifiers do
not move at all, and the support vector machine is identical to the decimal
place, which confirms no information was lost. So the mechanism is demonstrated,
not asserted.

Second, and more uncomfortably: two of the papers I cite show that this kind of
electrode error largely cancels when you subtract a reference frame, which is
what my pipeline does. If that were true here, my central result would be
inflated. So I made the cancelling fraction a parameter and swept it. The answer
holds across the whole range and only collapses when cancellation is essentially
perfect — which requires the electrodes to be in identical positions at both
measurements, and that is exactly the condition those same papers say fails in
practice.""")

# ═══════════════════════════════════════════════════════ 9 — limitations
s = slide()
heading(s, "What this is, and what it is not", kicker="Limitations")
rule(s)
tb(s, 0.85, 2.35, 5.4, 0.45, "What I can claim", size=19, bold=True, color=BLUE)
tb(s, 0.85, 2.95, 5.4, 3.2,
   "Which noise sources a training model needs, and why.\n\n"
   "That the conventional ablation protocol inverts that answer.\n\n"
   "A mechanism for the architecture result, tested rather than assumed.\n\n"
   "An open, reproducible pipeline.",
   size=17, line=1.3, space_after=4)
tb(s, 6.9, 2.35, 5.5, 0.45, "What I cannot", size=19, bold=True, color=ORANGE)
tb(s, 6.9, 2.95, 5.5, 3.2,
   "Every number here is from simulation. There is no hardware validation.\n\n"
   "Electrode bias is assumed independent per electrode; a stretched sheet would "
   "correlate it.\n\n"
   "76% is not deployable. Detecting that contact happened is reliable; "
   "identifying which touch it was is not.",
   size=17, color=MUTED, line=1.3, space_after=4)
num(s, 9)
note(s, """[6:03-6:43]  ~40s

I want to be straight about the boundaries.

What I can claim is which noise sources matter and why, that the conventional way
of testing that question gives the opposite answer, a tested mechanism for the
architecture result, and a pipeline anyone can rerun.

What I cannot claim is that any of this survives contact with hardware. Every
number is simulated. My electrode bias is drawn independently per electrode, and
on a hydrogel sheet under strain it would be spatially correlated — measuring that
covariance on a real sensor is the first experiment I would run next. And
seventy-six per cent is not a deployable figure. The per-class breakdown says
detecting that contact occurred is reliable; saying which kind of touch it was is
not.""")

# ═══════════════════════════════════════════════════════ 10 — close
s = slide()
tb(s, 0.85, 2.5, 11.6, 0.5, "IN ONE SENTENCE", size=14, bold=True, color=BLUE)
tb(s, 0.85, 3.15, 11.3, 2.2,
   "Noise-aware training closes most of the simulation-to-reality gap —\n"
   "but only if you test it against the world the device will actually meet.",
   size=30, bold=True, line=1.3)
tb(s, 0.85, 5.75, 11.3, 0.6, "Thank you. I'm happy to take questions.",
   size=20, color=MUTED)
num(s, 10)
note(s, """[6:43-6:55]  ~12s

If there is one thing to take away, it is this. Noise-aware training does close
most of the gap — but only if you evaluate it against the world the device will
actually meet, because the conventional alternative gives you the opposite
answer with complete confidence.

Thank you. I'm happy to take questions.""")

out = HERE / "EE52037_Piku_Presentation.pptx"
prs.save(out)
print("saved:", out)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
