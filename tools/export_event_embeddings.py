#!/usr/bin/env python3
"""Transform embeddings into the event attendee schema:
[
  {
    "id": "user_1",
    "name": "Tejas Nayak",
    "ticketId": "TKT-VIP-8841",
    "ticketType": "VIP All-Access",
    "avatar": "https://images.unsplash.com/photo-...",
    "vector": [0.0451, -0.0123, 0.0892, ...]
  }
]
"""
import json
import random
from pathlib import Path

# Seed for deterministic yet realistic output
random.seed(42)

TICKET_TYPES = [
    ("VIP All-Access", "VIP"),
    ("General Admission", "GA"),
    ("Speaker Pass", "SPK"),
    ("Early Bird Pass", "EB"),
    ("All-Access Pass", "ALL"),
    ("Press & Media", "MED"),
    ("Student Pass", "STU"),
    ("Executive VIP", "EXEC"),
]

AVATAR_POOL = [
    "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=150",
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150",
    "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=150",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=150",
    "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=150",
    "https://images.unsplash.com/photo-1528892952291-009c663ce843?w=150",
    "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=150",
    "https://images.unsplash.com/photo-1517462964-21fdcec3f25b?w=150",
    "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150",
]


def main():
    root = Path(__file__).resolve().parent.parent.parent
    src_file = root / "checkin" / "embeddings.json"

    with open(src_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    students = data.get("students", []) if isinstance(data, dict) else data

    # Backup existing booth-format file if not already backed up
    backup_file = root / "checkin" / "embeddings_booth.json"
    if not backup_file.exists() and isinstance(data, dict):
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Backed up booth format to {backup_file}")

    attendees = []
    for idx, s in enumerate(students):
        user_id = f"user_{idx + 1}"
        name = s.get("name", f"Attendee {idx + 1}")
        vector = s.get("embedding", s.get("vector", []))

        # Select ticket type and corresponding ticket ID
        ttype, tcode = random.choice(TICKET_TYPES)
        ticket_num = random.randint(1000, 9999)
        ticket_id = f"TKT-{tcode}-{ticket_num}"

        # Avatar: select from avatar pool
        avatar = random.choice(AVATAR_POOL)

        attendees.append({
            "id": user_id,
            "name": name,
            "ticketId": ticket_id,
            "ticketType": ttype,
            "avatar": avatar,
            "vector": vector,
        })

    # Save to checkin/embeddings.json and root embeddings.json
    out_checkin = root / "checkin" / "embeddings.json"
    out_root = root / "embeddings.json"

    print(f"Exporting {len(attendees)} records to {out_checkin}...")
    with open(out_checkin, "w", encoding="utf-8") as f:
        json.dump(attendees, f, indent=2)

    print(f"Exporting to {out_root}...")
    with open(out_root, "w", encoding="utf-8") as f:
        json.dump(attendees, f, indent=2)

    print("Export complete!")
    size_mb = out_checkin.stat().st_size / (1024 * 1024)
    print(f"Output file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
