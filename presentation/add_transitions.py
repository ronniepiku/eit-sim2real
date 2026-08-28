"""Add a uniform, subtle fade transition to every slide. Run after build_deck.py
and apply_script.py (must be last, since it edits the saved pptx directly)."""
from pathlib import Path
from pptx import Presentation
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn

HERE = Path(__file__).parent
PATH = HERE / "EE52037_Piku_Presentation.pptx"

TRANSITION_XML = (
    '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'spd="med"><p:fade/></p:transition>'
)

prs = Presentation(PATH)
n = 0
for s in prs.slides:
    sld = s._element
    # remove any existing transition first (idempotent re-runs)
    for existing in sld.findall(qn("p:transition")):
        sld.remove(existing)
    trans = parse_xml(TRANSITION_XML)
    # schema order: cSld, clrMapOvr?, transition?, timing? -- insert right after cSld
    cSld = sld.find(qn("p:cSld"))
    cSld.addnext(trans)
    n += 1

prs.save(PATH)
print(f"added fade transition to {n} slides")
