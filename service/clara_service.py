#!/usr/bin/env python3
"""
CLaRa Memory Service — FastAPI wrapper for CLaRa memory retrieval.

Runs on the NVIDIA machine, wraps the existing CLaRaMemorySystem.
Provides /retrieve (search + generate) and /store (ingest) endpoints.

Usage:
    cd /home/worker/workspace/ml-clara
    source venv/bin/activate
    python clara_service.py --device cuda:0 --port 8300 --quantization int8
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add ml-clara to path so we can import the memory system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# These will be imported after path setup in main()
memory_system = None
app = FastAPI(title="CLaRa Memory Service", version="0.1.0")

# CORS for local network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("clara-service")


# ── Request/Response Models ──────────────────────────────────────────


class RetrieveRequest(BaseModel):
    """Query CLaRa for relevant memories."""
    query: str = Field(..., description="User's message text to search against")
    top_k: int = Field(5, description="Number of memories to search")
    gen_top_k: int = Field(3, description="Number of memories to use for generation")
    max_tokens: int = Field(256, description="Max tokens for CLaRa generation")
    debug: bool = Field(True, description="Include debug info in response")


class StoreRequest(BaseModel):
    """Store a new message in CLaRa memory."""
    text: str = Field(..., description="Text to compress and store")
    role: str = Field("user", description="Message role (user/assistant)")
    session_id: str = Field("default", description="Session identifier")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class MemoryItem(BaseModel):
    """A single retrieved memory."""
    id: int
    text: str
    score: float
    timestamp: str
    metadata: Optional[Dict] = None


class DebugInfo(BaseModel):
    """Debug information for transparency."""
    query: str
    search_time_ms: float
    generate_time_ms: float
    total_memories: int
    searched: int
    used_for_generation: int
    memory_scores: List[Dict]


class RetrieveResponse(BaseModel):
    """Response from /retrieve — hybrid summary + memories."""
    summary: str = Field(..., description="CLaRa's generated reasoning about the memories")
    memories: List[MemoryItem] = Field(..., description="Retrieved memory items")
    formatted: str = Field(..., description="Pre-built context block for injection")
    has_memories: bool = Field(..., description="Whether any memories were found")
    debug: Optional[DebugInfo] = None


class StoreResponse(BaseModel):
    """Response from /store."""
    memory_id: int
    mse_loss: float
    shape: List[int]
    message: str


class StatsResponse(BaseModel):
    """Memory system statistics."""
    total_memories: int
    embedding_dimension: Optional[int]
    storage_directory: str
    compression_rate: int
    model_name: str


# ── Formatting ───────────────────────────────────────────────────────


def format_hybrid_output(
    summary: str,
    memories: List[Dict],
    query: str
) -> str:
    """
    Format the hybrid output: summary of context + listed memories.
    This is what gets injected into Claude's context.
    """
    if not memories:
        return ""

    lines = []
    lines.append("## Relevant Context from Long-Term Memory")
    lines.append("")
    lines.append(f"Based on your message, these past conversations appear relevant:")
    lines.append("")

    # CLaRa's generated summary (latent-space reasoning)
    if summary and summary != "No relevant memories found.":
        lines.append(f"**Context:** {summary}")
        lines.append("")

    # Individual memories
    lines.append("### Retrieved Memories")
    lines.append("")
    for i, mem in enumerate(memories, 1):
        score = mem.get("similarity_score", 0)
        timestamp = mem.get("timestamp", "unknown")
        # Parse timestamp to just date if possible
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            date_str = str(timestamp)

        text = mem.get("text", "")
        lines.append(f"**#{i}** (relevance: {score:.2f} | {date_str})")
        lines.append(f"> {text}")
        lines.append("")

    return "\n".join(lines)


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check."""
    global memory_system
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    stats = memory_system.get_stats()
    return {
        "status": "ok",
        "total_memories": stats["total_memories"],
        "model": stats["model_name"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """
    Query CLaRa for relevant memories.
    Returns hybrid format: generated summary + listed memories.
    """
    global memory_system
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    log.info(f"🔍 RETRIEVE query: {req.query[:100]}...")

    # Step 1: Search for relevant memories (FAISS similarity)
    t0 = time.time()
    search_results = memory_system.search_memories(req.query, top_k=req.top_k)
    search_time = (time.time() - t0) * 1000

    log.info(f"  Found {len(search_results)} memories in {search_time:.0f}ms")
    for i, mem in enumerate(search_results):
        log.info(f"  [{i+1}] score={mem['similarity_score']:.3f} | {mem['text'][:80]}...")

    if not search_results:
        return RetrieveResponse(
            summary="No relevant memories found.",
            memories=[],
            formatted="",
            has_memories=False,
            debug=DebugInfo(
                query=req.query,
                search_time_ms=search_time,
                generate_time_ms=0,
                total_memories=memory_system.get_stats()["total_memories"],
                searched=0,
                used_for_generation=0,
                memory_scores=[]
            ) if req.debug else None
        )

    # Step 2: Generate summary using CLaRa's decoder (latent-space reasoning)
    t1 = time.time()
    try:
        generated_text, _ = memory_system.retrieve_and_generate(
            query=req.query,
            top_k=req.gen_top_k,
            max_new_tokens=req.max_tokens
        )
    except Exception as e:
        log.error(f"  Generation failed: {e}")
        generated_text = f"(Generation error: {str(e)})"
    generate_time = (time.time() - t1) * 1000

    log.info(f"  Generated summary in {generate_time:.0f}ms: {generated_text[:100]}...")

    # Step 3: Format hybrid output
    formatted = format_hybrid_output(generated_text, search_results, req.query)

    # Build response (convert numpy types to native Python)
    memories = [
        MemoryItem(
            id=int(mem.get("memory_id", -1)),
            text=str(mem["text"]),
            score=float(mem["similarity_score"]),
            timestamp=str(mem.get("timestamp", "")),
            metadata=mem.get("custom_metadata")
        )
        for mem in search_results
    ]

    debug = None
    if req.debug:
        debug = DebugInfo(
            query=req.query,
            search_time_ms=round(float(search_time), 1),
            generate_time_ms=round(float(generate_time), 1),
            total_memories=int(memory_system.get_stats()["total_memories"]),
            searched=len(search_results),
            used_for_generation=min(req.gen_top_k, len(search_results)),
            memory_scores=[
                {"id": int(m.get("memory_id", -1)), "score": round(float(m["similarity_score"]), 4), "text_preview": m["text"][:60]}
                for m in search_results
            ]
        )

    log.info(f"  ✅ Returning {len(memories)} memories, formatted={len(formatted)} chars")

    return RetrieveResponse(
        summary=generated_text,
        memories=memories,
        formatted=formatted,
        has_memories=True,
        debug=debug
    )


@app.post("/store", response_model=StoreResponse)
async def store(req: StoreRequest):
    """Store a new message in CLaRa memory."""
    global memory_system
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    log.info(f"📦 STORE [{req.role}]: {req.text[:100]}...")

    # Format with role prefix
    formatted_text = f"{req.role}: {req.text}"

    metadata = req.metadata or {}
    metadata.update({
        "role": req.role,
        "session_id": req.session_id,
        "stored_at": datetime.now().isoformat()
    })

    memory_id, info = memory_system.add_memory(formatted_text, metadata)

    log.info(f"  ✅ Stored as memory #{memory_id}, MSE={info['mse_loss']:.4f}, shape={info['shape']}")

    return StoreResponse(
        memory_id=memory_id,
        mse_loss=info["mse_loss"],
        shape=info["shape"],
        message=f"Stored memory #{memory_id}"
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get memory system statistics."""
    global memory_system
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    s = memory_system.get_stats()
    return StatsResponse(**s)


@app.get("/memories")
async def list_memories(limit: int = 20, offset: int = 0):
    """List stored memories (for debugging)."""
    global memory_system
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    all_mems = memory_system.get_all_memories()
    # Sort by memory_id descending (newest first)
    all_mems.sort(key=lambda m: m.get("memory_id", 0), reverse=True)

    total = len(all_mems)
    page = all_mems[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "memories": [
            {
                "id": m.get("memory_id"),
                "text": m.get("text", "")[:200],
                "timestamp": m.get("timestamp"),
                "mse_loss": m.get("mse_loss"),
                "metadata": m.get("custom_metadata")
            }
            for m in page
        ]
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    global memory_system

    parser = argparse.ArgumentParser(description="CLaRa Memory Service")
    parser.add_argument("--port", type=int, default=8300, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--quantization", type=str, default="int8",
                        choices=["none", "int4", "int8"], help="Quantization mode")
    parser.add_argument("--model-path", type=str, default="./models/compression-128",
                        help="Path to CLaRa model")
    parser.add_argument("--storage-dir", type=str, default="./clara_service_storage",
                        help="Memory storage directory")
    parser.add_argument("--top-k", type=int, default=5, help="Default retrieval count")
    args = parser.parse_args()

    # Import memory system (needs ml-clara on path)
    from memory_system import CLaRaMemorySystem, MemoryConfig

    # Configure
    use_quant = args.quantization != "none"
    quant_bits = 8 if args.quantization == "int8" else 4

    config = MemoryConfig(
        model_name=args.model_path,
        storage_dir=args.storage_dir,
        compression_rate=128,
        top_k_retrieval=args.top_k,
        device=args.device,
        use_quantization=use_quant,
        quantization_bits=quant_bits if use_quant else 4
    )

    log.info("=" * 60)
    log.info("CLaRa Memory Service starting...")
    log.info(f"  Device:       {args.device}")
    log.info(f"  Quantization: {args.quantization}")
    log.info(f"  Model:        {args.model_path}")
    log.info(f"  Storage:      {args.storage_dir}")
    log.info(f"  Port:         {args.port}")
    log.info("=" * 60)

    # Initialize memory system
    memory_system = CLaRaMemorySystem(config)
    stats = memory_system.get_stats()
    log.info(f"✅ Memory system ready: {stats['total_memories']} existing memories")

    # Run server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
