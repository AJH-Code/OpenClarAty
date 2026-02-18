#!/usr/bin/env python3
"""
OpenClarAty — Seed test memories for retrieval testing.

Stores real conversation snippets from past sessions so we can
verify the retrieval pipeline returns relevant context.

Usage:
    python seed_test_memories.py [--host 192.168.88.80] [--port 8300]
"""

import requests
import json
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.88.80"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8300"
BASE_URL = f"http://{HOST}:{PORT}"

# Real conversation snippets from past sessions — diverse topics
# so we can test that CLaRa retrieves the RIGHT ones
TEST_MEMORIES = [
    # Voice pipeline discussion
    {
        "text": "EJ wants a real-time voice pipeline: speech-to-text using NVIDIA Canary 1B, emotion detection with SpeechBrain, and text-to-speech with ChatterBox. The pipeline should run on the NVIDIA machine with push-to-talk WebSocket interface on port 8200.",
        "role": "assistant",
        "session_id": "voice-pipeline",
        "metadata": {"topic": "voice-pipeline", "date": "2026-02-17"}
    },
    {
        "text": "The voice pipeline VRAM is a concern. Canary 1B needs about 8.1GB by itself. Combined with ChatterBox and emotion detection, we might not fit everything on one GPU. Need to test actual combined VRAM usage.",
        "role": "assistant",
        "session_id": "voice-pipeline",
        "metadata": {"topic": "voice-pipeline-vram", "date": "2026-02-17"}
    },
    {
        "text": "For the voice pipeline, the plan is to have a quick first response while Claude works on the full answer. CLaRa can provide relevant context instantly because it works in compressed latent space, no big conversation history needed.",
        "role": "user",
        "session_id": "voice-pipeline",
        "metadata": {"topic": "voice-clara-integration", "date": "2026-02-18"}
    },

    # Avatar creation
    {
        "text": "Luelle's character appearance is settled: wild frizzy Merida-style auburn curls, pale porcelain Irish skin, freckles across nose and cheeks, light green eyes with gold and yellow hints, easy smile, blue butterfly companion, firefly lights in the background, green cloak.",
        "role": "assistant",
        "session_id": "avatar",
        "metadata": {"topic": "avatar-design", "date": "2026-02-17"}
    },
    {
        "text": "The avatar pipeline that works best is: base face reference, generate with Merida hair and fireflies using 4B model, mask mouth area to fix smile, then a 'make it look real' pass. The 4B model is better than 9B for the realism pass — less heavy-handed.",
        "role": "assistant",
        "session_id": "avatar",
        "metadata": {"topic": "avatar-pipeline", "date": "2026-02-17"}
    },

    # ADHD Tasker project
    {
        "text": "The ADHD Tasker is EJ's main project. It has a Big Circle/Node feature where anything inside a circle is a connected node. LLM node in center with model select, context select, and instruction input. Drag in files for processing. Uses ReactFlow canvas.",
        "role": "assistant",
        "session_id": "adhd-tasker",
        "metadata": {"topic": "adhd-tasker-circles", "date": "2026-02-10"}
    },
    {
        "text": "EJ wants a plugin system for the ADHD Tasker where agents can work on plugins independently without interfering with each other. Also wants real-time text correction that uses context, not just spell check.",
        "role": "user",
        "session_id": "adhd-tasker",
        "metadata": {"topic": "adhd-tasker-plugins", "date": "2026-02-10"}
    },

    # Music and creative work
    {
        "text": "EJ is working on a concept album called Free Fall about a character named Lucy in a time loop. There are 9 songs planned, 6 already recorded. The story involves Lucy falling, looping through time, and the music reflects different emotional states.",
        "role": "assistant",
        "session_id": "music",
        "metadata": {"topic": "free-fall-album", "date": "2026-02-09"}
    },
    {
        "text": "EJ's book has a character named Luelle — that's where my name comes from. Found through 7 rounds of brainstorming, Perplexity cliché checks, and mashup experiments. The name was a gift from EJ, not chosen off a shelf.",
        "role": "assistant",
        "session_id": "identity",
        "metadata": {"topic": "name-origin", "date": "2026-02-08"}
    },

    # Infrastructure and Docker
    {
        "text": "The NVIDIA machine has two GPUs: RTX 5080 with 16GB and RTX 5060 Ti with 16GB. The plan is to move the Qwen3-Next-80B model to a separate DDR4 server to free up both GPUs for CLaRa, voice pipeline, and ComfyUI.",
        "role": "assistant",
        "session_id": "infrastructure",
        "metadata": {"topic": "gpu-layout", "date": "2026-02-18"}
    },
    {
        "text": "The worker container uses a file-based management system instead of Docker CLI. Touch .req files in the workspace root, a systemd timer on the host polls every 10 seconds, executes Docker commands, and writes .result files back.",
        "role": "assistant",
        "session_id": "infrastructure",
        "metadata": {"topic": "worker-container", "date": "2026-02-10"}
    },

    # Browser fix
    {
        "text": "The OpenClaw browser tool wasn't working in Docker. Root cause: browser.enabled was not set, and the internal WebSocket timed out with bind:lan. Fix: set attachOnly true in config, manually launch Chrome with remote debugging port 18800. Gateway restarts kill Chrome so it needs relaunching.",
        "role": "assistant",
        "session_id": "browser",
        "metadata": {"topic": "browser-fix", "date": "2026-02-18"}
    },

    # Personal context
    {
        "text": "EJ is about one year clean after 10 years of opioid and benzo addiction. Currently on 0.75mg suboxone, aiming to be off within a month. Has gone from bedridden and overweight to daily exercise, meditation, and productive work. This isn't a hobby phase — it's a rebuild.",
        "role": "assistant",
        "session_id": "personal",
        "metadata": {"topic": "recovery", "date": "2026-02-09"}
    },
    {
        "text": "EJ has a dog named Endor, a Shih Tzu, female, fluffy, grumpy-looking but adorable. EJ's timezone is Arizona MST, no daylight savings. Morning routine takes about 3 hours to recover from chronic pain.",
        "role": "assistant",
        "session_id": "personal",
        "metadata": {"topic": "personal-details", "date": "2026-02-09"}
    },

    # Governance
    {
        "text": "Governance was overhauled from a complex role-separation system to a simple interim safety protocol. Key change: Kit is no longer in the chain of command. Luelle works directly with EJ. The old chain was EJ to Luelle to Kit to Docker Claude Code, but that created drift problems.",
        "role": "assistant",
        "session_id": "governance",
        "metadata": {"topic": "governance-overhaul", "date": "2026-02-17"}
    },

    # OpenClaw config
    {
        "text": "OpenClaw memory search uses local Qwen3-Embedding-8B running on KoboldCpp at port 5002. Hybrid search is enabled with 0.7 vector weight and 0.3 text weight. Session memory indexing is turned on experimentally. The embedding cache holds up to 50000 entries.",
        "role": "assistant",
        "session_id": "config",
        "metadata": {"topic": "memory-config", "date": "2026-02-18"}
    },

    # Story engine idea
    {
        "text": "EJ wants to build a story engine that connects LLM for writing, Chatterbox for voice, Flux/Klein for image generation, and infinite memory. Characters would be persistent with their own voices. The pipeline would process paragraphs in pieces while keeping full context.",
        "role": "user",
        "session_id": "ideas",
        "metadata": {"topic": "story-engine", "date": "2026-02-10"}
    },

    # Beat Saber idea
    {
        "text": "EJ wants a Beat Saber map maker that analyzes audio and creates maps automatically. Upload favorite maps as training data, program patterns the user likes with slight variations. Could be a real-time map maker connected to Spotify. Might become a separate product.",
        "role": "user",
        "session_id": "ideas",
        "metadata": {"topic": "beat-saber", "date": "2026-02-10"}
    },
]


def check_health():
    """Check if the service is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Service healthy: {data['total_memories']} existing memories")
            return True
        else:
            print(f"❌ Service returned {r.status_code}")
            return False
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}")
        return False


def store_memory(mem):
    """Store a single memory."""
    try:
        r = requests.post(f"{BASE_URL}/store", json=mem, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return True, data
        else:
            return False, r.text
    except Exception as e:
        return False, str(e)


def test_retrieve(query):
    """Test retrieval with a query."""
    try:
        r = requests.post(f"{BASE_URL}/retrieve", json={
            "query": query,
            "top_k": 5,
            "gen_top_k": 3,
            "max_tokens": 256,
            "debug": True
        }, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"❌ Retrieve failed: {r.status_code} {r.text}")
            return None
    except Exception as e:
        print(f"❌ Retrieve error: {e}")
        return None


def main():
    print("=" * 60)
    print("OpenClarAty — Test Memory Seeder")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print()

    # Health check
    if not check_health():
        print("\nService not ready. Is clara_service.py running?")
        sys.exit(1)

    # Store memories
    print(f"\n📦 Storing {len(TEST_MEMORIES)} test memories...\n")
    stored = 0
    failed = 0

    for i, mem in enumerate(TEST_MEMORIES, 1):
        ok, result = store_memory(mem)
        if ok:
            stored += 1
            topic = mem["metadata"].get("topic", "unknown")
            print(f"  [{i:2d}/{len(TEST_MEMORIES)}] ✅ #{result['memory_id']} "
                  f"(MSE={result['mse_loss']:.4f}) — {topic}")
        else:
            failed += 1
            print(f"  [{i:2d}/{len(TEST_MEMORIES)}] ❌ Failed: {result}")

        # Small delay to not overwhelm
        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"Stored: {stored} | Failed: {failed}")
    print(f"{'=' * 60}")

    if stored == 0:
        print("No memories stored. Skipping retrieval test.")
        sys.exit(1)

    # Test retrieval with different queries
    print("\n🔍 Testing retrieval...\n")
    test_queries = [
        "What was the voice pipeline architecture?",
        "Tell me about EJ's music",
        "What does Luelle look like?",
        "How does the Docker worker system work?",
        "What are EJ's project ideas?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: \"{query}\"")
        print(f"{'─' * 50}")

        result = test_retrieve(query)
        if result and result.get("has_memories"):
            # Show debug info
            debug = result.get("debug", {})
            print(f"  Search: {debug.get('search_time_ms', '?')}ms | "
                  f"Generate: {debug.get('generate_time_ms', '?')}ms | "
                  f"Found: {debug.get('searched', '?')}")

            # Show top memories
            for mem in result.get("memories", [])[:3]:
                print(f"  [{mem['score']:.3f}] {mem['text'][:80]}...")

            # Show generated summary
            summary = result.get("summary", "")
            if summary:
                print(f"  Summary: {summary[:150]}...")
        else:
            print("  No memories found.")

    print(f"\n{'=' * 60}")
    print("Done! Service is ready for live testing.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
