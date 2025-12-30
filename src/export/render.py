import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_md(path, transcript, summary):
    md = []
    md.append("# Meeting Notes\n")
    md.append("## Short Summary\n")
    md.append(summary.get("short_summary", "") + "\n")

    def bullets(title, items):
        md.append(f"## {title}\n")
        for x in items or []:
            md.append(f"- {x}")
        md.append("")

    bullets("Key Points", summary.get("key_points"))
    bullets("Decisions", summary.get("decisions"))
    bullets("Risks / Issues", summary.get("risks_or_issues"))

    md.append("## Action Items\n")
    for a in summary.get("action_items") or []:
        md.append(f"- **Task:** {a.get('task','')}")
        md.append(f"  - Owner: {a.get('owner', None)}")
        md.append(f"  - Due: {a.get('due_date', None)}")
        md.append(f"  - Evidence: {a.get('evidence_timestamp', None)}")
    md.append("")

    md.append("## Highlights\n")
    for h in summary.get("highlights") or []:
        ts = h.get("timestamp")
        text = h.get("text", "")
        if ts:
            md.append(f"- [{ts}] {text}")
        else:
            md.append(f"- {text}")
    md.append("")

    md.append("## Transcript\n")
    md.append(transcript)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def save_pdf(path, md_text):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    y = height - 40
    for line in md_text.splitlines():
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:120])
        y -= 14
    c.save()
