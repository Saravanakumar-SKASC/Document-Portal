"""
Generate FICTIONAL airline documents for demos and evaluation.

"Demo Air" is an invented airline. Every page is stamped as fictional and not for
operational use; the procedures, limits and part numbers are made up and must never
be used on a real aircraft. Contact details are fake and exist only to demonstrate
PII redaction.

Run:  python scripts/make_demo_docs.py      (writes PDFs to samples/)
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "samples"
BANNER = "DEMO AIR (FICTIONAL AIRLINE) - FOR DEMONSTRATION ONLY - NOT FOR OPERATIONAL USE"

BAGGAGE_REV12 = [
    """Demo Air Baggage Policy - Revision 12 (effective 01 March 2026)

1. Checked baggage allowance
Economy: 1 piece, maximum 23 kg.
Premium Economy: 2 pieces, maximum 23 kg each.
Business: 2 pieces, maximum 32 kg each.
Excess baggage is charged at EUR 25 per additional kg on short-haul flights.""",
    """2. Cabin baggage
Each passenger may carry 1 cabin bag up to 7 kg and 56 x 36 x 23 cm, plus 1 personal item.

3. Lithium batteries and power banks
Power banks up to 100 Wh are permitted in cabin baggage only, maximum 2 per passenger.
Power banks may be used on board to charge personal devices.""",
]

BAGGAGE_REV13 = [
    """Demo Air Baggage Policy - Revision 13 (effective 01 September 2026)

1. Checked baggage allowance
Economy: 1 piece, maximum 25 kg.
Premium Economy: 2 pieces, maximum 23 kg each.
Business: 2 pieces, maximum 32 kg each.
Excess baggage is charged at EUR 25 per additional kg on short-haul flights.""",
    """2. Cabin baggage
Each passenger may carry 1 cabin bag up to 7 kg and 56 x 36 x 23 cm, plus 1 personal item.

3. Lithium batteries and power banks
Power banks up to 100 Wh are permitted in cabin baggage only, maximum 2 per passenger.
Power banks must NOT be used or charged on board at any time and must stay in the seat pocket
or under the seat, not in the overhead locker.""",
]

TASK_CARD = [
    """Demo Air Maintenance Task Card - ATA 32 Landing Gear
Task 32-41-00-210-801: Main landing gear brake wear inspection
Aircraft type: DA-350 (fictional). Revision: Rev 7.

WARNING: Make sure the parking brake is set and wheel chocks are installed before this task.

1. Apply brake pressure of 3000 psi using the parking brake.
2. Measure the wear pin protrusion on each brake assembly P/N 2315-0045-001.
3. If the wear pin is flush with the housing or below it, the brake has reached its wear limit.
   Replace the brake assembly P/N 2315-0045-001 before the next flight.""",
    """4. If the wear pin protrudes 1 mm or more, the brake is serviceable. Record the value in the tech log.
5. Wheel nut torque after brake replacement: 150 ft-lb, then back off and re-torque to 90 ft-lb.

Related task: 32-42-00-710-802 (brake operational test after replacement).
Tooling: wear pin gauge T-3241-06.

Maintenance planning contact: j.murphy@demoair.example, phone +353 1 555 0147.""",
]

CREW_SOP = [
    """Demo Air Cabin Crew SOP - Chapter 5 Turbulence (Rev 4)

5.1 Light turbulence: The seat belt sign is on. Crew check that passengers are seated,
secure carts and continue service with care.

5.2 Moderate turbulence: Stop the service immediately, secure carts and galley equipment,
and crew take the nearest seat and fasten their harness.

5.3 Severe turbulence: Crew sit down immediately wherever they are and secure themselves.
No service or cabin checks until the flight deck announces it is safe.""",
    """Chapter 6 Medical Events (Rev 4)

6.1 For any unwell passenger, inform the flight deck and ask for medical professionals on board.
6.2 The Emergency Medical Kit (EMK) may only be opened by a doctor or nurse.
6.3 The AED is stored in the forward overhead locker at row 1.

Sample incident report (fictional): passenger Jane Example, Passport No: K12345678,
HKID A123456(7), contact jane.example@mail.example, card 4111 1111 1111 1111 refunded.""",
]

DOCS = {
    "demoair_baggage_policy_rev12.pdf": BAGGAGE_REV12,
    "demoair_baggage_policy_rev13.pdf": BAGGAGE_REV13,
    "demoair_task_card_ata32_brakes.pdf": TASK_CARD,
    "demoair_cabin_crew_sop.pdf": CREW_SOP,
}


def write_pdf(path: Path, pages: list[str]):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((40, 30), BANNER, fontsize=8, color=(0.7, 0, 0))
        page.insert_textbox(pymupdf.Rect(40, 50, 560, 800), text, fontsize=10.5)
    doc.save(path)
    doc.close()


def main():
    OUT.mkdir(exist_ok=True)
    for name, pages in DOCS.items():
        write_pdf(OUT / name, pages)
        print(f"wrote samples/{name}")


if __name__ == "__main__":
    main()
