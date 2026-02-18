/**
 * OpenClaw Memory (CLaRa + RAG) Plugin
 *
 * Dual memory retrieval:
 * 1. CLaRa — compressed latent reasoning (Mistral-7B LoRA + FAISS)
 * 2. RAG — standard embedding search (Qwen3-Embedding-8B + SQLite vec)
 *
 * Results are merged: if both systems find the same memory (matched by
 * timestamp), the score is boosted. This covers blind spots in both systems.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createRequire } from "node:module";
import path from "node:path";
import os from "node:os";

// ============================================================================
// Config
// ============================================================================

interface PluginConfig {
  endpoint: string;
  autoRecall: boolean;
  topK: number;
  minScore: number;
  timeoutMs: number;
  classify: {
    enabled: boolean;
    timeoutMs: number;
  };
  rag: {
    enabled: boolean;
    embeddingUrl: string;
    embeddingModel: string;
    dbPath: string;
    topK: number;
  };
}

const DEFAULT_CONFIG: PluginConfig = {
  endpoint: "http://192.168.88.80:8300",
  autoRecall: true,
  topK: 3,
  minScore: 0.15,
  timeoutMs: 3000,
  classify: {
    enabled: true,
    timeoutMs: 1500,
  },
  rag: {
    enabled: true,
    embeddingUrl: "http://192.168.88.80:5002/v1/embeddings",
    embeddingModel: "Qwen3-Embedding-8B-Q8_0",
    dbPath: path.join(os.homedir(), ".openclaw", "memory", "main.sqlite"),
    topK: 3,
  },
};

// ============================================================================
// Skip filters — don't waste queries on filler
// ============================================================================

const MIN_WORDS = 3;

const SKIP_PHRASES = new Set([
  // Confirmations
  "ok", "okay", "k", "kk", "sure", "yes", "no", "yep", "yep yep", "nope",
  "yea", "yeah", "nah", "done", "k done", "ok done", "got it", "sounds good",
  "go ahead", "do it", "okey dokey", "okey doke",
  // Gratitude
  "thanks", "thank you", "ty", "thx",
  // Reactions
  "cool", "nice", "lol", "lmao", "haha", "hmm", "huh", "wow", "damn", "bruh",
  "interesting", "good", "good good", "great", "perfect", "fine", "alright",
  "bet", "word", "dope", "sick", "right", "true", "fair", "agreed", "same",
  // Questions (too vague)
  "what", "why", "how", "when", "really", "watcha doin", "stuck",
  // Greetings
  "hey", "hey you", "hi", "hello", "sup", "yo",
  // Status
  "restarted", "finished", "crashed", "done",
  // Navigation
  "continue", "go on", "keep going", "next", "one sec", "hold on", "wait",
  // Corrections
  "nevermind", "nvm", "my bad", "oops", "whoops",
  // Emoticons
  ":d", ":)", ":(", ":p", "<3", "xd",
]);

function shouldSkipQuery(text: string): boolean {
  let cleaned = text;
  cleaned = cleaned.replace(/^\[.*?\]\s*/s, "");
  if (cleaned.includes("Conversation info")) {
    cleaned = cleaned.replace(/Conversation info.*?```\s*/s, "");
  }
  cleaned = cleaned.trim();

  const words = cleaned.split(/\s+/).filter((w) => w.length > 0);
  if (words.length < MIN_WORDS) return true;

  const normalized = cleaned.toLowerCase().replace(/[.!?,;:'"()\-\[\]{}]+/g, "").replace(/\s+/g, " ").trim();
  if (SKIP_PHRASES.has(normalized)) return true;

  return false;
}

// ============================================================================
// Mistral classifier — smart filler detection using the already-loaded model
// ============================================================================

interface ClassifyResponse {
  classification: "FILLER" | "SEARCH";
  raw_output: string;
  time_ms: number;
}

async function classifyMessage(
  endpoint: string, text: string, timeoutMs: number,
): Promise<ClassifyResponse | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(`${endpoint}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, max_tokens: 3 }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) return null;
    return (await response.json()) as ClassifyResponse;
  } catch {
    return null; // On error, fall through to retrieval (fail open)
  }
}

// ============================================================================
// CLaRa retrieval
// ============================================================================

interface ClaraMemory {
  id: number;
  text: string;
  score: number;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

interface ClaraResponse {
  summary: string;
  memories: ClaraMemory[];
  has_memories: boolean;
  debug?: { search_time_ms: number; generate_time_ms: number; total_memories: number };
}

async function queryCLaRa(
  endpoint: string, query: string, topK: number, timeoutMs: number,
): Promise<ClaraResponse | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(`${endpoint}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) return null;
    return (await response.json()) as ClaraResponse;
  } catch {
    return null;
  }
}

// ============================================================================
// RAG retrieval (Qwen3 embeddings + SQLite vec)
// ============================================================================

interface RagResult {
  text: string;
  path: string;
  score: number;
  startLine: number;
  endLine: number;
}

async function getEmbedding(
  url: string, model: string, text: string, timeoutMs: number,
): Promise<number[] | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: text, model }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) return null;
    const data = (await response.json()) as { data: Array<{ embedding: number[] }> };
    return data.data[0]?.embedding ?? null;
  } catch {
    return null;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Cached chunk store — loads embeddings once, refreshes every 10 minutes.
 */
interface CachedChunk {
  text: string;
  path: string;
  startLine: number;
  endLine: number;
  vec: number[];
}

let chunkCache: CachedChunk[] | null = null;
let cacheLoadedAt = 0;
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes

function loadChunkCache(dbPath: string): CachedChunk[] {
  const now = Date.now();
  if (chunkCache && now - cacheLoadedAt < CACHE_TTL_MS) {
    return chunkCache;
  }

  const require = createRequire(import.meta.url);
  const { DatabaseSync } = require("node:sqlite");
  const db = new DatabaseSync(dbPath, { readOnly: true });

  const rows = db.prepare(
    "SELECT text, path, start_line, end_line, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 2000"
  ).all() as Array<{ text: string; path: string; start_line: number; end_line: number; embedding: string }>;

  db.close();

  const chunks: CachedChunk[] = [];
  for (const row of rows) {
    try {
      chunks.push({
        text: row.text,
        path: row.path,
        startLine: row.start_line,
        endLine: row.end_line,
        vec: JSON.parse(row.embedding),
      });
    } catch {
      // Skip bad embeddings
    }
  }

  chunkCache = chunks;
  cacheLoadedAt = now;
  return chunks;
}

/**
 * Search the OpenClaw memory SQLite DB using embeddings.
 * Uses in-memory cache (refreshes every 10 min).
 */
async function queryRAG(
  dbPath: string, embeddingUrl: string, embeddingModel: string,
  query: string, topK: number, timeoutMs: number,
): Promise<RagResult[]> {
  try {
    // Get query embedding
    const queryVec = await getEmbedding(embeddingUrl, embeddingModel, query, timeoutMs);
    if (!queryVec) return [];

    // Load chunks (cached)
    const chunks = loadChunkCache(dbPath);

    // Score each chunk
    const scored: RagResult[] = [];
    for (const chunk of chunks) {
      const score = cosineSimilarity(queryVec, chunk.vec);
      scored.push({
        text: chunk.text,
        path: chunk.path,
        score,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
      });
    }

    // Sort by score, return top K
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  } catch {
    return [];
  }
}

// ============================================================================
// Result merging
// ============================================================================

interface MergedMemory {
  text: string;
  score: number;
  source: "clara" | "rag" | "both";
  claraScore?: number;
  ragScore?: number;
}

function extractTimestamp(text: string): string | null {
  // Match [YYYY-MM-DD HH:MM UTC ...] patterns
  const match = text.match(/\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})/);
  return match ? match[1] : null;
}

function mergeResults(
  claraMemories: ClaraMemory[],
  ragResults: RagResult[],
  minScore: number,
): MergedMemory[] {
  const merged: MergedMemory[] = [];
  const ragUsed = new Set<number>();

  // Process CLaRa results
  for (const cm of claraMemories) {
    if (cm.score < minScore) continue;

    const claraTs = extractTimestamp(cm.text);
    let matchedRag: RagResult | null = null;
    let matchedIdx = -1;

    // Check if RAG found the same memory (by timestamp match)
    if (claraTs) {
      for (let i = 0; i < ragResults.length; i++) {
        if (ragUsed.has(i)) continue;
        const ragTs = extractTimestamp(ragResults[i].text);
        if (ragTs && ragTs === claraTs) {
          matchedRag = ragResults[i];
          matchedIdx = i;
          break;
        }
      }
    }

    if (matchedRag && matchedIdx >= 0) {
      // Both systems found it — boost score
      ragUsed.add(matchedIdx);
      merged.push({
        text: cm.text,
        score: Math.max(cm.score, matchedRag.score) * 1.3, // 30% boost
        source: "both",
        claraScore: cm.score,
        ragScore: matchedRag.score,
      });
    } else {
      merged.push({
        text: cm.text,
        score: cm.score,
        source: "clara",
        claraScore: cm.score,
      });
    }
  }

  // Add RAG-only results
  for (let i = 0; i < ragResults.length; i++) {
    if (ragUsed.has(i)) continue;
    const rr = ragResults[i];
    if (rr.score < minScore) continue;
    merged.push({
      text: rr.text,
      score: rr.score,
      source: "rag",
      ragScore: rr.score,
    });
  }

  // Sort by score descending
  merged.sort((a, b) => b.score - a.score);
  return merged;
}

// ============================================================================
// Plugin
// ============================================================================

const claraPlugin = {
  id: "memory-clara",
  name: "Memory (CLaRa + RAG)",
  description: "Dual memory retrieval — CLaRa latent reasoning + Qwen3 embedding RAG",

  register(api: OpenClawPluginApi) {
    const cfg: PluginConfig = {
      ...DEFAULT_CONFIG,
      ...(api.pluginConfig as Partial<PluginConfig>),
      classify: { ...DEFAULT_CONFIG.classify, ...((api.pluginConfig as any)?.classify ?? {}) },
      rag: { ...DEFAULT_CONFIG.rag, ...((api.pluginConfig as any)?.rag ?? {}) },
    };

    api.logger.info(
      `memory-clara: registered (clara: ${cfg.endpoint}, classify: ${cfg.classify.enabled ? "enabled" : "disabled"}, rag: ${cfg.rag.enabled ? "enabled" : "disabled"})`,
    );

    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event) => {
        if (!event.prompt || event.prompt.length < 10) return;

        const prompt = event.prompt;
        if (shouldSkipQuery(prompt)) return;

        // Skip heartbeats, system events, cron
        if (
          prompt.includes("HEARTBEAT") ||
          prompt.includes("Read HEARTBEAT.md") ||
          prompt.includes("[cron:") ||
          prompt.includes("NO_REPLY") ||
          prompt.startsWith("System:")
        ) {
          return;
        }

        // Smart classifier — use Mistral-7B to decide if message is worth searching
        if (cfg.classify.enabled) {
          // Extract just the user's actual message text for classification
          let msgText = prompt;
          // Strip Telegram metadata headers
          msgText = msgText.replace(/^\[.*?\]\s*/s, "");
          if (msgText.includes("Conversation info")) {
            msgText = msgText.replace(/Conversation info.*?```\s*/s, "");
          }
          msgText = msgText.trim();

          // Only classify messages that aren't too long (long = likely has content)
          if (msgText.length < 200) {
            const classifyResult = await classifyMessage(
              cfg.endpoint, msgText, cfg.classify.timeoutMs,
            );
            if (classifyResult?.classification === "FILLER") {
              api.logger.info?.(
                `memory-clara: skipped (classifier: FILLER, ${classifyResult.time_ms.toFixed(0)}ms) "${msgText.slice(0, 60)}"`,
              );
              return;
            }
            // SEARCH or null (error/timeout) → proceed with retrieval
            if (classifyResult) {
              api.logger.info?.(
                `memory-clara: classifier: ${classifyResult.classification} (${classifyResult.time_ms.toFixed(0)}ms)`,
              );
            }
          }
        }

        // Query both systems in parallel
        const [claraResult, ragResults] = await Promise.all([
          queryCLaRa(cfg.endpoint, prompt, cfg.topK, cfg.timeoutMs),
          cfg.rag.enabled
            ? queryRAG(cfg.rag.dbPath, cfg.rag.embeddingUrl, cfg.rag.embeddingModel, prompt, cfg.rag.topK, cfg.timeoutMs)
            : Promise.resolve([]),
        ]);

        const claraMemories = claraResult?.memories ?? [];

        // Merge and rank
        const merged = mergeResults(claraMemories, ragResults, cfg.minScore);
        if (merged.length === 0) return;

        // Take top results (max 5)
        const top = merged.slice(0, 5);

        const memoryContext = top
          .map((m) => {
            const tag = m.source === "both" ? "🔗" : m.source === "clara" ? "🧠" : "📎";
            return `${tag} ${m.text.slice(0, 300)}${m.text.length > 300 ? "..." : ""}`;
          })
          .join("\n\n");

        const claraSummary = claraResult?.summary ?? "";
        const claraMs = claraResult?.debug?.search_time_ms?.toFixed(0) ?? "?";
        const claraGen = claraResult?.debug?.generate_time_ms?.toFixed(0) ?? "?";
        const sources = top.map((m) => m.source);
        const bothCount = sources.filter((s) => s === "both").length;
        const claraOnly = sources.filter((s) => s === "clara").length;
        const ragOnly = sources.filter((s) => s === "rag").length;

        api.logger.info?.(
          `memory-clara: injecting ${top.length} memories (clara: ${claraOnly}, rag: ${ragOnly}, both: ${bothCount}, search: ${claraMs}ms)`,
        );

        return {
          prependContext: `<clara-memories>
These memories were automatically retrieved from past conversations based on the user's message. They may or may not be relevant — use your judgment. If relevant, incorporate naturally without mentioning this system. If not relevant, ignore them entirely.

Legend: 🔗 = found by both systems (high confidence), 🧠 = CLaRa latent memory, 📎 = RAG embedding match

Summary: ${claraSummary}

${memoryContext}
</clara-memories>`,
        };
      });
    }

    // Service
    api.registerService({
      id: "memory-clara",
      async start() {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 2000);
          const response = await fetch(`${cfg.endpoint}/health`, { signal: controller.signal });
          clearTimeout(timeout);
          if (response.ok) {
            api.logger.info("memory-clara: CLaRa service healthy ✅");
          } else {
            api.logger.warn("memory-clara: CLaRa not reachable");
          }
        } catch {
          api.logger.warn("memory-clara: CLaRa not reachable (will retry on queries)");
        }
      },
      stop() {
        api.logger.info("memory-clara: stopped");
      },
    });
  },
};

export default claraPlugin;
