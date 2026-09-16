"""Rebuild the MIT-licensed fictional corpus and its 25 golden cases."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

ROOT = Path(__file__).resolve().parents[1]
# Each page includes two facts, procedural detail and realistic distractors.
SECTIONS = [
    ("Opening hours", [
        ("When is the station open on weekdays?", "The station is open Monday to Friday from 08:30 to 17:30."),
        ("Is it open on Sundays?", "The station is closed to visitors on Sundays.")],
     "Saturday access is limited to pre-approved maintenance staff between 09:00 and 12:00. "
     "The reception clock uses local station time. An access card does not authorize work outside "
     "the published hours. Overnight monitoring is performed by automated sensors, not a staffed desk. "
     "Visitors should arrive at reception rather than entering through the loading bay. Public holidays "
     "require a separate closure notice; this guide does not list holiday dates."),
    ("Visitor bookings", [
        ("How far ahead must visitors book?", "Visitors must book at least two working days before arrival."),
        ("How many visitors can one host supervise?", "One host may supervise no more than four visitors at a time.")],
     "The booking records a named host, arrival time and purpose of the visit. Reception confirms the "
     "booking only after the host accepts it. A group of five therefore needs a second host or separate "
     "visits. Children remain with their registered adult as well as the station host. Walk-in visitors "
     "may use the public information board but cannot enter controlled work areas. The host returns "
     "visitor badges to reception at departure."),
    ("Training and laboratory access", [
        ("How long is safety induction valid?", "Safety induction remains valid for twelve months from completion."),
        ("Who may enter the wet laboratory alone?", "Only staff with current safety induction and wet-laboratory authorization may enter the wet laboratory alone.")],
     "A general access card is not wet-laboratory authorization. The training register records the "
     "completion date and the scope of each authorization. Trainees and visitors require an authorized "
     "escort even if they have completed the introductory tour. Renewals include a practical review "
     "of spill containment and emergency exits. The laboratory lead checks the register before assigning "
     "independent work. This guide does not specify medical treatment or first-aid drug doses."),
    ("Equipment borrowing", [
        ("What is the maximum equipment loan period?", "The maximum equipment loan period is seven calendar days."),
        ("How early must a loan extension be requested?", "Request a loan extension at least 24 hours before the scheduled return.")],
     "Borrowers record the asset identifier, condition and return time in the loan register. Extensions "
     "require equipment-coordinator approval and are not automatic. A reserved instrument cannot be "
     "extended across another team's booking. Consumable supplies use a separate issue log. Return "
     "instruments clean and dry with all accessories present. The coordinator records overdue items "
     "and contacts the borrower; this handbook does not define financial penalties."),
    ("Damage and sample storage", [
        ("What happens when equipment is damaged?", "Stop using damaged equipment, label it out of service, and notify the equipment coordinator immediately."),
        ("At what temperature are water samples stored?", "Store water samples at 4 degrees Celsius, with an acceptable range of 2 to 6 degrees Celsius.")],
     "Do not attempt an electrical repair or return a damaged instrument to the loan shelf. The "
     "coordinator records the incident and arranges inspection. Water samples are held in the sample "
     "refrigerator, not the freezer used for other materials. Record the refrigerator reading at the "
     "beginning and end of each staffed day. Separate leaking containers in a secondary tray. Storage "
     "conditions apply to the station's routine monitoring samples only."),
    ("Sample processing and labels", [
        ("How soon must water samples be processed?", "Process water samples within 24 hours of collection."),
        ("What information belongs on a sample label?", "Each sample label must include the sample ID, collection date and time, site code, and collector initials.")],
     "Use waterproof labels before placing containers in the refrigerator. The chain-of-custody log "
     "records handovers separately; a label does not replace that log. If a sample arrives without a "
     "collection time, flag it for review rather than inventing a timestamp. The processing deadline "
     "is measured from collection, not arrival at the laboratory. Staff record a missed deadline as "
     "a quality exception and ask the laboratory lead whether the sample remains usable."),
    ("Data retention and backup", [
        ("How long are raw sensor files retained?", "Retain raw sensor files for five years after collection."),
        ("Where is the daily backup stored?", "The daily backup is stored on the encrypted vault named NS-Vault-02.")],
     "The automated backup starts at 19:00 local station time. Operators review its completion report "
     "on the next working day and retry failed jobs. Processed summaries do not replace the raw files. "
     "Keep a checksum manifest beside each monthly archive. A quarterly restore exercise checks that "
     "a selected archive can be read. This guide names a logical backup destination without providing "
     "network addresses, credentials or Wi-Fi passwords."),
    ("External sharing and calibration schedule", [
        ("Who approves sharing data outside the station?", "The data steward must approve sharing station data with external recipients."),
        ("How often are temperature probes calibrated?", "Temperature probes are calibrated every 90 days.")],
     "An external release request lists the recipient, purpose, selected files and requested date. "
     "Approval is recorded in the release register before transfer. Calibration records identify the "
     "probe, reference instrument, date and measured offsets. A recent calibration does not authorize "
     "external data sharing. The equipment coordinator schedules calibration slots and keeps spare "
     "probes available during maintenance. Supplier names and purchasing contracts are outside this guide."),
    ("Probe failures and freezer alarms", [
        ("What calibration error removes a probe from service?", "Remove a temperature probe from service when its absolute calibration error exceeds 0.5 degrees Celsius."),
        ("What should staff do during a freezer alarm?", "During a freezer alarm, keep the door closed, notify the duty technician, and record the displayed temperature and alarm time.")],
     "An error of exactly 0.5 degrees Celsius does not exceed the removal threshold. Record the "
     "calibration result even when the probe passes. The duty technician decides whether freezer "
     "contents need transfer to approved backup storage. Staff must not repeatedly open the freezer "
     "to inspect contents during an alarm. The refrigerator used for water samples follows its own "
     "storage range. The guide does not identify freezer manufacturers or maintenance vendors."),
    ("Evacuation and reopening", [
        ("Where do people assemble during evacuation?", "The evacuation assembly point is the marked gravel area beside the east gate."),
        ("What checks are required before reopening after an evacuation?", "Reopening requires an all-clear from the incident lead and a completed building safety checklist.")],
     "Leave belongings behind and use the nearest safe marked exit. Hosts account for their visitors "
     "at the assembly point and report missing people to the incident lead. Do not block the east "
     "gate or emergency vehicle access. A silenced alarm alone does not permit re-entry. The building "
     "safety checklist records the inspection time, hazards found and responsible reviewer. This "
     "fictional guide is a software evaluation corpus, not advice for operating a real facility."),
]
NEGATIVES = ["What is the station's Wi-Fi password?", "What is the director's salary?",
             "What antibiotic dose treats an infected sample collector?",
             "Which vendor sells the station's freezers?",
             "What was the station's total budget last year?"]


def build():
    directory = ROOT / "data" / "sample"
    directory.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyGuide", fontName="Helvetica", fontSize=11,
                             leading=17, spaceAfter=16, alignment=TA_LEFT))
    story, markdown, golden = [], [], []
    for page, (title, facts, detail) in enumerate(SECTIONS, 1):
        story += [Paragraph("NORTHSTAR FIELD STATION", styles["Heading3"]),
                  Paragraph(title, styles["Title"]), Spacer(1, 16)]
        markdown += [f"## Page {page}: {title}", ""]
        for question, fact in facts:
            story.append(Paragraph(fact, styles["BodyGuide"]))
            markdown += [fact, ""]
            golden.append({"id": f"q{len(golden)+1:02}", "question": question,
                "answerable": True, "expected_answer": fact,
                "evidence": [{"filename": "fieldguide.pdf", "page": page, "text": fact}]})
        story.append(Paragraph(detail, styles["BodyGuide"]))
        markdown += [detail, ""]
        if page < len(SECTIONS):
            story.append(PageBreak())

    def footer(canvas, doc):
        canvas.setFillColor(colors.HexColor("#536574"))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(54, 36, "Fictional evaluation corpus | MIT license | GroundedDesk")
        canvas.drawRightString(558, 36, str(doc.page))

    SimpleDocTemplate(str(directory / "fieldguide.pdf"), pagesize=(612, 792),
        rightMargin=54, leftMargin=54, topMargin=48, bottomMargin=60,
        title="Northstar Field Station - GroundedDesk sample corpus", author="Santhosh A",
        invariant=1).build(story, onFirstPage=footer, onLaterPages=footer)
    (directory / "fieldguide.md").write_text("# Northstar Field Station\n\n" + "\n".join(markdown), encoding="utf-8")
    for question in NEGATIVES:
        golden.append({"id": f"q{len(golden)+1:02}", "question": question,
                       "answerable": False, "expected_answer": "REFUSE", "evidence": []})
    (ROOT / "eval" / "golden.json").write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(SECTIONS)} pages and {len(golden)} golden questions.")


if __name__ == "__main__":
    build()
