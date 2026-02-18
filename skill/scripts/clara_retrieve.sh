#!/bin/bash
# CLaRa Memory Retrieval — queries the CLaRa service for relevant memories.
#
# Usage: clara_retrieve.sh "<user message>"
#
# Returns formatted context block for injection into conversation.
# Also prints debug info to stderr for transparency.

CLARA_HOST="${CLARA_HOST:-192.168.88.80}"
CLARA_PORT="${CLARA_PORT:-8300}"
CLARA_URL="http://${CLARA_HOST}:${CLARA_PORT}"

QUERY="$1"

if [ -z "$QUERY" ]; then
    echo "Error: No query provided"
    exit 1
fi

# Check if service is up
if ! curl -sf "${CLARA_URL}/health" > /dev/null 2>&1; then
    echo "⚠️ CLaRa service not reachable at ${CLARA_URL}"
    exit 0  # Don't fail hard — just skip memory retrieval
fi

# Query CLaRa
RESPONSE=$(curl -sf -X POST "${CLARA_URL}/retrieve" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg q "$QUERY" '{
        query: $q,
        top_k: 5,
        gen_top_k: 3,
        max_tokens: 256,
        debug: true
    }')" 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$RESPONSE" ]; then
    echo "⚠️ CLaRa query failed"
    exit 0
fi

# Check if memories were found
HAS_MEMORIES=$(echo "$RESPONSE" | jq -r '.has_memories')

if [ "$HAS_MEMORIES" = "false" ]; then
    # Print debug to stderr
    echo "🔍 CLaRa: No relevant memories found for query" >&2
    echo ""
    exit 0
fi

# Print debug info to stderr (visible in logs, not in model output)
echo "═══ CLaRa Debug ═══" >&2
echo "$RESPONSE" | jq -r '.debug | "Query: \(.query[:80])...\nSearch: \(.search_time_ms)ms | Generate: \(.generate_time_ms)ms\nMemories found: \(.searched) | Used for gen: \(.used_for_generation)"' >&2
echo "$RESPONSE" | jq -r '.debug.memory_scores[] | "  [\(.id)] score=\(.score) | \(.text_preview)"' >&2
echo "═══════════════════" >&2

# Output the formatted context block (this is what the model sees)
echo "$RESPONSE" | jq -r '.formatted'
