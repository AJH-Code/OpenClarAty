---
name: clara-memory
description: Query CLaRa infinite memory for relevant context from past conversations. Use on every inbound user message to retrieve compressed memories.
metadata: {"openclaw": {"emoji": "🧠", "requires": {"bins": ["curl", "jq"]}}}
---

# CLaRa Memory Retrieval

CLaRa (Continuous Latent Reasoning and Attention) compresses past conversations into latent-space memory tokens. When a user sends a message, query CLaRa to retrieve relevant historical context.

## When to Use

**On EVERY inbound user message**, run the retrieval script BEFORE formulating your response. This gives you context from past conversations that may be relevant.

**IMPORTANT RULES:**
- Only query on USER messages — never on your own output
- If the service is down, skip gracefully and respond normally
- If no memories are found, respond normally (don't mention the empty search)
- If memories ARE found, incorporate them naturally — don't say "according to CLaRa" or "my memory system shows"

## How to Use

Run the retrieval script with the user's message:

```bash
bash {baseDir}/scripts/clara_retrieve.sh "the user's message here"
```

The script returns a formatted context block with:
1. **Summary** — CLaRa's latent-space reasoning about why these memories are relevant
2. **Retrieved Memories** — the actual past conversation snippets with relevance scores

Use this context naturally in your response. You don't need to reference every memory — just let it inform your understanding.

## Service Details

- **Host:** 192.168.88.80 (NVIDIA machine)
- **Port:** 8300
- **Health check:** `curl http://192.168.88.80:8300/health`

## Example Flow

1. User says: "Hey, what was that voice pipeline architecture we discussed?"
2. You run: `bash {baseDir}/scripts/clara_retrieve.sh "Hey, what was that voice pipeline architecture we discussed?"`
3. CLaRa returns relevant memories about the voice pipeline discussion
4. You respond incorporating that context naturally

## Debug Output

The script prints debug info to stderr (search times, scores, memory previews). This appears in logs but not in the response. When testing, check stderr for:
- Search/generate timing
- Number of memories found
- Relevance scores per memory
- Text previews of matched memories
