# Task: Memory Brain Visualizer Web App

Build a web application that visualizes an AI memory system. This connects to a CLaRa memory service API.

## API Endpoint
- **Base URL:** `http://localhost:8300`
- **GET /memories?limit=1000&offset=0** — returns all memories with: id, text (first 200 chars), timestamp, mse_loss, metadata
- **GET /stats** — returns total_memories, model_name, etc.
- **GET /health** — health check

## What to Build

A single-page web app (vanilla HTML/CSS/JS or lightweight framework like Three.js for 3D) with:

### 1. Timeline View (Main Screen)
- Left sidebar: list of days that have memories (extracted from timestamps)
- Click a day → center panel shows all memories for that day in chronological order
- Each memory shows: timestamp, truncated text, emotion tags (if present in metadata), topic (if present)

### 2. Memory Detail View
- Click any memory → opens a detail panel/modal
- Shows full text, timestamp, all metadata
- Shows connected nodes as a visual graph:
  - **Emotions node** — shows emotion tags associated with this memory
  - **Related Memories node** — shows other memories linked to this one (via metadata.related_memories array)
  - **Topic node** — shows the topic category
- Click on an emotion → shows all other memories tagged with that emotion
- Click on a related memory → navigates to that memory's detail view

### 3. 3D Brain Background (the cool part)
- A translucent 3D brain model in the background (can use a simple sphere-based representation or find a brain mesh)
- Different regions of the brain map to different emotions/categories:
  - Frontal: reasoning, planning
  - Temporal: language, memory recall
  - Limbic: emotions
  - Occipital: visual/creative
- When a memory is selected, the relevant brain region lights up/glows
- When retrieval happens, show lines/particles flowing between brain regions
- Use Three.js or similar for the 3D rendering

### 4. Search/Filter
- Search bar at top to filter memories by text
- Filter by emotion, topic, date range
- Results highlight in the timeline and light up corresponding brain regions

## Technical Requirements
- Serve on port 8400
- Use a simple HTTP server (python http.server or Node.js express)
- All API calls go to `http://localhost:8300` (the CLaRa service running on this machine)
- Mobile-friendly is not required — desktop browser is fine
- Dark theme (the brain looks better against dark)

## Memory Data Format
Each memory from the API looks like:
```json
{
  "id": 42,
  "text": "user: What was the name of EJ's dog?\nassistant: Endor...",
  "timestamp": "2026-02-18T05:30:00",
  "mse_loss": 0.0123,
  "metadata": {
    "role": "user",
    "session_id": "abc123",
    "stored_at": "2026-02-18T05:30:00",
    "emotions": ["curiosity"],
    "topic": "personal info",
    "related_memories": [15, 88, 201]
  }
}
```

Note: emotions, topic, and related_memories may not exist yet on all memories — handle gracefully (show "not tagged" or similar).

## Output
- Put everything in `/home/worker/workspace/brain-viewer/`
- Include a start script or instructions to run it
- The app should work when opened in a browser at `http://192.168.88.80:8400`
