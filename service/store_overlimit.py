#!/usr/bin/env python3
"""
Store the 34 over-limit exchanges by stripping noise and tighter truncation.
Most are 801-953 chars — barely over. Telegram headers alone are 60-80 chars of waste.
"""

import json
import sys
import time
import re
import requests
from datetime import datetime

CLARA_URL = "http://192.168.88.80:8300"
MAX_CHARS = 790  # safely under 800
INPUT_FILE = "/tmp/over-limit-full.json"


def strip_telegram_header(text):
    """Remove [Telegram EJ (@EJ_message) id:... UTC] prefix."""
    text = re.sub(r'\[Telegram EJ \(@EJ_message\) id:\d+ \+\d+[sm] \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]\s*', '', text)
    # Also strip [message_id: ...] lines
    text = re.sub(r'\[message_id: [^\]]+\]\s*', '', text)
    # Strip queued message headers
    text = re.sub(r'---\s*Queued #\d+\s*', '', text)
    text = re.sub(r'\[Queued messages while agent was busy\]\s*---\s*', '', text)
    return text.strip()


def truncate_assistant_tight(text, max_chars=250):
    """Get first 1-2 sentences, tighter than the original."""
    # Take first sentence or two
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    for s in sentences[:2]:
        if len(result) + len(s) > max_chars:
            break
        result = result + " " + s if result else s
    return result or text[:max_chars]


def format_timestamp_tag(timestamp, session_id):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except:
        date_str = timestamp[:19] if timestamp else "unknown"
    return f"[{date_str} | session:{session_id[:8]}]"


def store_memory(text, session_id, timestamp):
    data = {
        "text": text,
        "role": "exchange",
        "session_id": session_id,
        "metadata": {
            "source": "session_ingest_v3_compressed",
            "session_id": session_id,
            "timestamp": timestamp,
            "was_overlimit": True,
        }
    }
    try:
        r = requests.post(f"{CLARA_URL}/store", json=data, timeout=30)
        return r.status_code == 200, r.json() if r.status_code == 200 else r.text
    except Exception as e:
        return False, str(e)


def main():
    dry_run = "--dry-run" in sys.argv

    with open(INPUT_FILE) as f:
        exchanges = json.load(f)

    print(f"{'=' * 60}")
    print(f"OpenClarAty — Store Over-Limit (smart compression)")
    print(f"{'=' * 60}")
    print(f"Exchanges: {len(exchanges)}")
    if dry_run:
        print("MODE: DRY RUN")
    print()

    if not dry_run:
        try:
            r = requests.get(f"{CLARA_URL}/health", timeout=5)
            print(f"✅ CLaRa: {r.json()['total_memories']} memories\n")
        except:
            print("❌ CLaRa not reachable")
            sys.exit(1)

    stored = 0
    still_over = 0

    for i, ex in enumerate(exchanges, 1):
        tag = format_timestamp_tag(ex["timestamp"], ex["session_id"])
        
        # Clean user text
        user_clean = strip_telegram_header(ex["user"])
        if len(user_clean) > 450:
            user_clean = user_clean[:447] + "..."

        # Start with tight assistant truncation
        # Budget: MAX_CHARS - tag(~45) - "user: "(6) - "\nassistant: "(13) - user_len
        tag_overhead = len(tag) + 1  # tag + newline
        user_part = f"user: {user_clean}"
        budget_for_assistant = MAX_CHARS - tag_overhead - len(user_part) - len("\nassistant: ")
        
        if budget_for_assistant < 50:
            # User message is too long, trim it more
            user_clean = user_clean[:350] + "..."
            user_part = f"user: {user_clean}"
            budget_for_assistant = MAX_CHARS - tag_overhead - len(user_part) - len("\nassistant: ")

        assistant_short = truncate_assistant_tight(ex["assistant"], max_chars=max(budget_for_assistant, 100))

        window = f"{tag}\n{user_part}\nassistant: {assistant_short}"

        # Final safety check
        if len(window) > 800:
            # Hard truncate
            window = window[:797] + "..."
            still_over += 1

        print(f"  [{i:2d}/{len(exchanges)}] {ex['window_len']}→{len(window)} chars", end="", flush=True)

        if dry_run:
            print(f" [DRY]")
            if "--verbose" in sys.argv:
                print(f"    {window[:150]}...")
            continue

        ok, result = store_memory(window, ex["session_id"], ex["timestamp"])
        if ok:
            stored += 1
            print(f" ✅")
        else:
            print(f" ❌ {str(result)[:60]}")

        time.sleep(0.3)

    print(f"\n{'=' * 60}")
    print(f"Stored: {stored} | Hard-truncated: {still_over}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
