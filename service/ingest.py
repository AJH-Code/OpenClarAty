#!/usr/bin/env python3
"""
OpenClarAty — Session ingestion v3

Strategy:
  - User message: keep full text (summarize only if >256 tokens)
  - Assistant response: first paragraph/couple sentences only
  - Tag with session_id + timestamp for lookup
  - Manifest tracks what's been ingested

Result: most exchanges fit in ~256 tokens without any summarization.
CLaRa acts as an index — if more detail needed, pull the full exchange
from the session file using the timestamp.
"""

import json
import os
import sys
import glob
import time
import re
import requests
from datetime import datetime

CLARA_URL = os.environ.get("CLARA_URL", "http://192.168.88.80:8300")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest-manifest.json")
MAX_CHARS = 800  # ~256 tokens
BATCH_DELAY = 0.3

SKIP_PATTERNS = [
    "Read HEARTBEAT.md if it exists",
    "HEARTBEAT_OK",
    "Pre-compaction memory flush",
    "A new session was started via /new",
    "[cron:",
    "Exec completed (",
    "NO_REPLY",
]


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"ingested": {}, "version": 3}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def should_skip(text):
    for p in SKIP_PATTERNS:
        if p in text:
            return True
    return False


def clean_message(text):
    """Strip metadata, timestamps from message text."""
    # Strip conversation info blocks
    if "Conversation info (untrusted metadata):" in text:
        lines = text.split("\n")
        actual = []
        in_json = False
        past_meta = False
        for ln in lines:
            if ln.strip() == "```json":
                in_json = True
                continue
            if ln.strip() == "```" and in_json:
                in_json = False
                past_meta = True
                continue
            if in_json:
                continue
            if past_meta:
                actual.append(ln)
        text = "\n".join(actual).strip()

    # Strip timestamp prefix
    if text.startswith("[") and "] " in text[:40]:
        bracket_end = text.index("] ")
        text = text[bracket_end + 2:]

    # Strip replied message blocks
    if "Replied message (untrusted, for context):" in text:
        lines = text.split("\n")
        actual = []
        in_json = False
        skip_block = False
        for ln in lines:
            if "Replied message" in ln:
                skip_block = True
                continue
            if skip_block and ln.strip() == "```json":
                in_json = True
                continue
            if in_json and ln.strip() == "```":
                in_json = False
                skip_block = False
                continue
            if in_json or skip_block:
                continue
            actual.append(ln)
        text = "\n".join(actual).strip()

    return text.strip()


def truncate_assistant(text):
    """Get first paragraph or first ~2 sentences of assistant response."""
    # Split on double newline (paragraphs)
    paragraphs = text.split("\n\n")
    first = paragraphs[0].strip()

    # If first paragraph is short enough, use it
    if len(first) <= 400:
        return first

    # Otherwise take first 2 sentences
    sentences = re.split(r'(?<=[.!?])\s+', first)
    result = ""
    for s in sentences[:3]:
        if len(result) + len(s) > 400:
            break
        result = result + " " + s if result else s

    return result or first[:400]


def format_timestamp_tag(timestamp, session_id):
    """Create a location tag for the memory."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except:
        date_str = timestamp[:19] if timestamp else "unknown"
    return f"[{date_str} | session:{session_id[:8]}]"


def extract_exchanges(filepath):
    """Extract user+assistant exchange pairs."""
    session_id = os.path.basename(filepath).replace(".jsonl", "")
    exchanges = []
    current_user = None
    current_user_ts = None
    current_assistant_parts = []
    last_timestamp = None

    with open(filepath) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "message":
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            timestamp = entry.get("timestamp", "")
            content = msg.get("content", [])

            texts = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))

            full_text = "\n".join(texts).strip()
            if not full_text or should_skip(full_text):
                continue

            full_text = clean_message(full_text)
            if not full_text or len(full_text) < 20:
                continue

            if role == "user":
                # Save previous exchange
                if current_user and current_assistant_parts:
                    exchanges.append({
                        "user": current_user,
                        "assistant": "\n".join(current_assistant_parts),
                        "user_timestamp": current_user_ts,
                        "timestamp": last_timestamp,
                    })
                current_user = full_text
                current_user_ts = timestamp
                current_assistant_parts = []
                last_timestamp = timestamp
            elif role == "assistant" and current_user:
                current_assistant_parts.append(full_text)
                last_timestamp = timestamp

    if current_user and current_assistant_parts:
        exchanges.append({
            "user": current_user,
            "assistant": "\n".join(current_assistant_parts),
            "user_timestamp": current_user_ts,
            "timestamp": last_timestamp,
        })

    return exchanges, session_id


def build_window(exchange, session_id):
    """Build a storable window: full user + truncated assistant + location tag."""
    user_text = exchange["user"]
    assistant_text = exchange["assistant"]
    timestamp = exchange.get("user_timestamp", exchange["timestamp"])

    # Location tag
    tag = format_timestamp_tag(timestamp, session_id)

    # User: keep full, but if absurdly long (>500 chars), take key portion
    if len(user_text) > 500:
        # Flag for potential summarization
        user_text = user_text[:500] + "..."

    # Assistant: first paragraph/sentences only
    assistant_short = truncate_assistant(assistant_text)

    window = f"{tag}\nuser: {user_text}\nassistant: {assistant_short}"
    return window


def store_memory(text, session_id, timestamp, metadata=None):
    """Store a memory in CLaRa."""
    data = {
        "text": text,
        "role": "exchange",
        "session_id": session_id,
        "metadata": {
            "source": "session_ingest_v3",
            "session_id": session_id,
            "timestamp": timestamp,
            **(metadata or {}),
        }
    }
    try:
        r = requests.post(f"{CLARA_URL}/store", json=data, timeout=30)
        if r.status_code == 200:
            return True, r.json()
        return False, r.text
    except Exception as e:
        return False, str(e)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session", help="Single session file to ingest")
    p.add_argument("--sessions-dir", default=os.path.expanduser("~/.openclaw/agents/main/sessions"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--show-windows", action="store_true", help="Print each window")
    p.add_argument("--stats-only", action="store_true", help="Just show size distribution")
    args = p.parse_args()

    manifest = load_manifest()

    if args.session:
        files = [args.session]
    else:
        files = sorted(glob.glob(os.path.join(args.sessions_dir, "*.jsonl")))
        if args.limit:
            files = files[:args.limit]

    print(f"{'=' * 60}")
    print(f"OpenClarAty — Session Ingestion v3")
    print(f"{'=' * 60}")
    print(f"Strategy: full user + truncated assistant + location tag")
    print(f"Target: ~256 tokens (~{MAX_CHARS} chars) per window")
    print(f"Files: {len(files)}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print()

    if not args.dry_run and not args.stats_only:
        try:
            r = requests.get(f"{CLARA_URL}/health", timeout=5)
            data = r.json()
            print(f"✅ CLaRa: {data['total_memories']} memories\n")
        except:
            print("❌ CLaRa not reachable")
            sys.exit(1)

    total_stored = 0
    total_over = 0
    total_under = 0
    size_buckets = {"<200": 0, "200-400": 0, "400-600": 0, "600-800": 0, ">800": 0}

    for fi, filepath in enumerate(files, 1):
        session_id = os.path.basename(filepath).replace(".jsonl", "")
        short_id = session_id[:12]

        if session_id in manifest["ingested"] and not args.force:
            continue

        exchanges, sid = extract_exchanges(filepath)
        if not exchanges:
            continue

        stored_this = 0
        over_this = 0

        for ex in exchanges:
            window = build_window(ex, session_id)
            wlen = len(window)

            # Stats
            if wlen < 200: size_buckets["<200"] += 1
            elif wlen < 400: size_buckets["200-400"] += 1
            elif wlen < 600: size_buckets["400-600"] += 1
            elif wlen < 800: size_buckets["600-800"] += 1
            else: size_buckets[">800"] += 1

            if wlen > MAX_CHARS:
                over_this += 1
                total_over += 1
                if args.show_windows:
                    print(f"  ⚠️  OVER ({wlen}): {window[:100]}...")
                continue

            total_under += 1

            if args.show_windows:
                print(f"  ✅ ({wlen}): {window[:100]}...")

            if args.stats_only or args.dry_run:
                stored_this += 1
                continue

            ok, result = store_memory(window, session_id, ex["timestamp"])
            if ok:
                stored_this += 1
                print(f"    stored {stored_this}/{len(exchanges)} ...", end="\r", flush=True)
            else:
                print(f"  ❌ {result[:80]}")
            time.sleep(BATCH_DELAY)

        total_stored += stored_this

        if stored_this > 0 or over_this > 0:
            print(f"  [{fi:3d}/{len(files)}] {short_id}... "
                  f"fit:{stored_this} over:{over_this} total:{len(exchanges)}    ")

        if not args.dry_run and not args.stats_only and stored_this > 0:
            manifest["ingested"][session_id] = {
                "timestamp": datetime.utcnow().isoformat(),
                "stored": stored_this,
                "skipped_over": over_this,
                "total_exchanges": len(exchanges),
            }
            save_manifest(manifest)

    print(f"\n{'=' * 60}")
    print(f"Fit under limit: {total_under}")
    print(f"Over limit:      {total_over}")
    print(f"{'=' * 60}")
    print(f"\nSize distribution:")
    for bucket, count in size_buckets.items():
        bar = "█" * (count // 2)
        print(f"  {bucket:>8s}: {count:4d} {bar}")
    print()


if __name__ == "__main__":
    main()
