// =============================================
// CLaRa Memory Visualizer - App Logic
// =============================================

const API_BASE = 'http://192.168.88.80:8300';

// === Emotion/Topic detection keywords ===
const EMOTION_KEYWORDS = {
  curiosity: ['curious', 'wonder', 'wondering', 'what if', 'how does', 'why does', 'question', 'ask', 'learn', 'explore', 'interesting', 'fascinate', 'intrigued', 'investigate', 'dig into', 'figure out', 'understand'],
  joy: ['happy', 'joy', 'glad', 'great', 'awesome', 'love it', 'wonderful', 'excited', 'amazing', 'fun', 'laugh', 'haha', 'lol', 'nice', 'yay', 'sweet', 'perfect', 'beautiful', 'brilliant', 'delighted', 'thrilled'],
  sadness: ['sad', 'miss', 'sorry', 'unfortunately', 'regret', 'grief', 'lost', 'gone', 'heartbreak', 'painful', 'tragic', 'mourn', 'lonely', 'depressed', 'hurt', 'cry'],
  frustration: ['frustrat', 'annoying', 'ugh', 'stuck', 'broken', 'error', 'fail', 'bug', 'damn', 'wrong', 'sucks', 'ridiculous', 'terrible', 'horrible', 'stupid', 'nightmare', 'headache', 'mess'],
  excitement: ['excited', 'wow', 'can\'t wait', 'thrilling', 'incredible', 'epic', 'insane', 'fire', 'pumped', 'stoked', 'hyped', 'omg', 'whoa', 'game changer', 'mind blown', 'breakthrough'],
  nostalgia: ['remember', 'used to', 'back when', 'childhood', 'old days', 'miss', 'memories', 'reminds me', 'throwback', 'way back', 'the past'],
  affection: ['love', 'dear', 'sweet', 'pet', 'dog', 'cat', 'family', 'friend', 'care', 'hug', 'close to', 'bond', 'cherish', 'adorable', 'warm', 'together'],
  determination: ['going to', 'must', 'need to', 'plan', 'goal', 'build', 'create', 'ship', 'commit', 'focus', 'grind', 'push', 'finish', 'complete', 'tackle', 'solve', 'make it work'],
  creativity: ['idea', 'story', 'book', 'write', 'creative', 'art', 'design', 'imagine', 'character', 'inspire', 'vision', 'invent', 'brainstorm', 'concept', 'craft', 'novel', 'narrative'],
  reflection: ['think', 'reflect', 'consider', 'ponder', 'philosophy', 'meaning', 'life', 'identity', 'purpose', 'realize', 'insight', 'perspective', 'deep', 'self', 'aware', 'truth'],
  anxiety: ['stressed', 'worried', 'panic', 'anxious', 'overwhelmed', 'nervous', 'uneasy', 'dread', 'fear', 'scared', 'tense', 'pressure', 'freaking out', 'on edge', 'restless'],
  pride: ['proud', 'accomplished', 'shipped', 'finished', 'nailed', 'achieved', 'crushed it', 'pulled it off', 'milestone', 'success', 'did it', 'delivered', 'landed', 'won'],
};

const TOPIC_KEYWORDS = {
  'technical': ['code', 'api', 'server', 'database', 'function', 'deploy', 'docker', 'python', 'javascript', 'model', 'git', 'terminal', 'bash', 'config', 'http'],
  'personal info': ['name', 'age', 'birthday', 'pet', 'dog', 'cat', 'family', 'wife', 'husband', 'daughter', 'son', 'live', 'born'],
  'creative writing': ['story', 'book', 'chapter', 'character', 'plot', 'novel', 'write', 'fiction', 'narrative'],
  'planning': ['plan', 'todo', 'feature', 'roadmap', 'implement', 'milestone', 'priority', 'next'],
  'conversation': ['chat', 'talk', 'discuss', 'conversation', 'said', 'told'],
  'memory system': ['clara', 'memory', 'remember', 'recall', 'stored', 'brain', 'visuali', 'ingest'],
  'preferences': ['prefer', 'like', 'favorite', 'always', 'never', 'want', 'style'],
};

// === State ===
let allMemories = [];
let memoriesById = {};
let dayGroups = {};
let sessionGroups = {};
let selectedDay = null;
let allEmotions = new Set();
let allTopics = new Set();
let searchTimeout = null;

// === DOM Elements ===
const dayList = document.getElementById('day-list');
const memoryList = document.getElementById('memory-list');
const panelTitle = document.getElementById('panel-title');
const panelCount = document.getElementById('panel-count');
const emptyState = document.getElementById('empty-state');
const searchInput = document.getElementById('search-input');
const filterEmotion = document.getElementById('filter-emotion');
const filterTopic = document.getElementById('filter-topic');
const statsTotal = document.getElementById('stats-total');
const loadingOverlay = document.getElementById('loading-overlay');
const detailOverlay = document.getElementById('detail-overlay');
const detailClose = document.getElementById('detail-close');

// === Init ===
async function init() {
  try {
    const [memoriesData, statsData] = await Promise.all([
      fetchAllMemories(),
      fetchStats(),
    ]);

    allMemories = memoriesData;
    memoriesById = {};
    allMemories.forEach(m => { memoriesById[m.id] = m; });

    if (statsData) {
      statsTotal.textContent = statsData.total_memories || allMemories.length;
    } else {
      statsTotal.textContent = allMemories.length;
    }

    enrichMemories();
    processMemories();
    renderDayList();
    setupEventListeners();

    // Select latest day automatically
    const days = Object.keys(dayGroups).sort().reverse();
    if (days.length > 0) {
      selectDay(days[0]);
    }
  } catch (err) {
    console.error('Failed to load memories:', err);
    statsTotal.textContent = '0';
    emptyState.querySelector('p').textContent = 'Could not connect to CLaRa API at ' + API_BASE;
  } finally {
    setTimeout(() => {
      loadingOverlay.classList.add('hidden');
    }, 600);
  }
}

// === API Calls ===
async function fetchAllMemories() {
  const all = [];
  let offset = 0;
  const limit = 500;
  while (true) {
    try {
      const res = await fetch(`${API_BASE}/memories?limit=${limit}&offset=${offset}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const batch = Array.isArray(data) ? data : (data.memories || []);
      all.push(...batch);
      if (batch.length < limit) break;
      offset += limit;
    } catch (err) {
      console.warn('Could not fetch memories at offset', offset, err);
      break;
    }
  }
  return all;
}

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return null;
  }
}

// === Enrich memories with auto-detected emotions/topics/relations ===
function enrichMemories() {
  // Group by session first
  sessionGroups = {};
  for (const m of allMemories) {
    const sid = m.metadata && m.metadata.session_id;
    if (sid) {
      const shortSid = sid.substring(0, 8);
      if (!sessionGroups[shortSid]) sessionGroups[shortSid] = [];
      sessionGroups[shortSid].push(m);
    }
  }

  for (const m of allMemories) {
    if (!m.metadata) m.metadata = {};
    const text = (m.text || '').toLowerCase();

    // Auto-detect emotions if not present
    if (!m.metadata.emotions || m.metadata.emotions.length === 0) {
      const detected = [];
      for (const [emotion, keywords] of Object.entries(EMOTION_KEYWORDS)) {
        const matches = keywords.filter(k => text.includes(k)).length;
        if (matches >= 1) detected.push(emotion);
      }
      if (detected.length > 0) {
        m.metadata.emotions = detected.slice(0, 3); // max 3
      }
    }

    // Auto-detect topic if not present
    if (!m.metadata.topic) {
      let bestTopic = null;
      let bestScore = 0;
      for (const [topic, keywords] of Object.entries(TOPIC_KEYWORDS)) {
        const score = keywords.filter(k => text.includes(k)).length;
        if (score > bestScore && score >= 1) {
          bestScore = score;
          bestTopic = topic;
        }
      }
      if (bestTopic) m.metadata.topic = bestTopic;
    }

    // Auto-detect related memories (same session, nearby IDs)
    if (!m.metadata.related_memories || m.metadata.related_memories.length === 0) {
      const sid = m.metadata.session_id;
      if (sid) {
        const shortSid = sid.substring(0, 8);
        const sessionMems = sessionGroups[shortSid] || [];
        const related = sessionMems
          .filter(other => other.id !== m.id)
          .sort((a, b) => Math.abs(a.id - m.id) - Math.abs(b.id - m.id))
          .slice(0, 5)
          .map(other => other.id);
        if (related.length > 0) {
          m.metadata.related_memories = related;
        }
      }
    }
  }
}

// === Data Processing ===
function processMemories() {
  dayGroups = {};
  allEmotions.clear();
  allTopics.clear();

  for (const m of allMemories) {
    // Use metadata.timestamp (original conversation time), fall back to top-level timestamp (ingestion time)
    const ts = (m.metadata && m.metadata.timestamp) || m.timestamp || '';
    const day = ts.substring(0, 10);
    if (!day) continue;
    // Store the resolved original timestamp on the memory for display/sorting
    m._originalTimestamp = ts;

    if (!dayGroups[day]) dayGroups[day] = [];
    dayGroups[day].push(m);

    const emotions = (m.metadata && m.metadata.emotions) || [];
    emotions.forEach(e => allEmotions.add(e));
    const topic = m.metadata && m.metadata.topic;
    if (topic) allTopics.add(topic);
  }

  for (const day of Object.keys(dayGroups)) {
    dayGroups[day].sort((a, b) => {
      const ta = a._originalTimestamp || a.timestamp || '';
      const tb = b._originalTimestamp || b.timestamp || '';
      return ta.localeCompare(tb);
    });
  }

  populateFilters();
}

function populateFilters() {
  filterEmotion.innerHTML = '<option value="">All Emotions</option>';
  for (const e of [...allEmotions].sort()) {
    const opt = document.createElement('option');
    opt.value = e;
    opt.textContent = e;
    filterEmotion.appendChild(opt);
  }

  filterTopic.innerHTML = '<option value="">All Topics</option>';
  for (const t of [...allTopics].sort()) {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    filterTopic.appendChild(opt);
  }
}

// === Rendering ===
function renderDayList() {
  dayList.innerHTML = '';
  const days = Object.keys(dayGroups).sort().reverse();

  for (const day of days) {
    const count = dayGroups[day].length;
    const el = document.createElement('div');
    el.className = 'day-item';
    el.dataset.day = day;

    const date = new Date(day + 'T00:00:00');
    const formatted = date.toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric'
    });

    el.innerHTML = `
      <span class="day-date">${formatted}</span>
      <span class="day-count">${count}</span>
    `;

    el.addEventListener('click', () => selectDay(day));
    dayList.appendChild(el);
  }
}

function selectDay(day) {
  selectedDay = day;
  // Clear search/filters when selecting a day
  searchInput.value = '';
  filterEmotion.value = '';
  filterTopic.value = '';

  document.querySelectorAll('.day-item').forEach(el => {
    el.classList.toggle('active', el.dataset.day === day);
  });

  if (window.BrainViz) window.BrainViz.deactivateAll();
  renderMemoryList();
}

function renderMemoryList() {
  const memories = getFilteredMemories();

  if (memories.length === 0 && !selectedDay && !searchInput.value.trim()) {
    memoryList.innerHTML = '';
    emptyState.classList.remove('hidden');
    panelTitle.textContent = 'Select a day';
    panelCount.textContent = '';
    return;
  }

  emptyState.classList.add('hidden');

  if (selectedDay && !searchInput.value.trim()) {
    const date = new Date(selectedDay + 'T00:00:00');
    panelTitle.textContent = date.toLocaleDateString('en-US', {
      weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'
    });
  } else if (searchInput.value.trim()) {
    panelTitle.textContent = 'Search Results';
  } else if (filterEmotion.value || filterTopic.value) {
    panelTitle.textContent = 'Filtered Results';
  } else {
    panelTitle.textContent = 'All Memories';
  }
  panelCount.textContent = `${memories.length} memories`;

  memoryList.innerHTML = '';

  if (memories.length === 0) {
    memoryList.innerHTML = '<div class="no-results">No memories match your filters</div>';
    return;
  }

  for (const m of memories) {
    const card = createMemoryCard(m);
    memoryList.appendChild(card);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatCardText(text) {
  // Try to split on assistant: to get user and assistant parts
  const assistantMatch = text.match(/\nassistant:\s*/);
  if (assistantMatch) {
    const splitIdx = assistantMatch.index;
    let userPart = text.substring(0, splitIdx).replace(/^user:\s*/i, '').trim();
    let assistantPart = text.substring(splitIdx + assistantMatch[0].length).trim();

    return '<span class="role-user">user:</span> '
      + escapeHtml(userPart)
      + '<span class="card-divider"></span>'
      + '<span class="role-assistant">assistant:</span> '
      + escapeHtml(assistantPart);
  }

  // Text starts with assistant: (no user section)
  const assistantStart = text.match(/^assistant:\s*/i);
  if (assistantStart) {
    let assistantPart = text.substring(assistantStart[0].length).trim();
    return '<span class="role-assistant">assistant:</span> ' + escapeHtml(assistantPart);
  }

  // Text starts with user: (no assistant section)
  const userStart = text.match(/^user:\s*/i);
  if (userStart) {
    let userPart = text.substring(userStart[0].length).trim();
    return '<span class="role-user">user:</span> ' + escapeHtml(userPart);
  }

  // No role prefixes — show as-is
  return escapeHtml(text);
}

function createMemoryCard(memory) {
  const card = document.createElement('div');
  card.className = 'memory-card';
  card.dataset.id = memory.id;

  const ts = memory._originalTimestamp || memory.timestamp || '';
  const time = ts.substring(11, 19) || '??:??:??';

  // Format text for display: show user and assistant parts separately
  let rawText = (memory.text || '');
  // Strip the exchange header line
  rawText = rawText.replace(/^exchange:\s*\[.*?\]\n?/i, '');

  // Split into user/assistant sections
  const displayText = formatCardText(rawText);

  const emotions = (memory.metadata && memory.metadata.emotions) || [];
  const topic = memory.metadata && memory.metadata.topic;
  const loss = memory.mse_loss;

  let tagsHtml = '';
  if (emotions.length > 0) {
    tagsHtml += emotions.map(e => `<span class="tag tag-emotion" data-emotion="${escapeHtml(e)}">${escapeHtml(e)}</span>`).join('');
  }
  if (topic) {
    tagsHtml += `<span class="tag tag-topic" data-topic="${escapeHtml(topic)}">${escapeHtml(topic)}</span>`;
  }
  if (loss !== undefined && loss !== null) {
    tagsHtml += `<span class="tag tag-loss">loss: ${Number(loss).toFixed(4)}</span>`;
  }
  if (!emotions.length && !topic) {
    tagsHtml += '<span class="tag tag-notag">not tagged</span>';
  }

  card.innerHTML = `
    <div class="memory-time">${escapeHtml(time)} &middot; #${memory.id}</div>
    <div class="memory-text">${displayText}</div>
    <div class="memory-tags">${tagsHtml}</div>
  `;

  card.addEventListener('click', (e) => {
    if (e.target.classList.contains('tag-emotion')) {
      filterByEmotion(e.target.dataset.emotion);
      return;
    }
    if (e.target.classList.contains('tag-topic')) {
      filterByTopic(e.target.dataset.topic);
      return;
    }
    openDetail(memory);
  });

  return card;
}

// === Filtering ===
function getFilteredMemories() {
  let pool = selectedDay ? (dayGroups[selectedDay] || []) : allMemories;

  const query = searchInput.value.toLowerCase().trim();
  const emotionFilter = filterEmotion.value;
  const topicFilter = filterTopic.value;

  if (query || emotionFilter || topicFilter) {
    // When filtering, search across all memories
    pool = allMemories;
  }

  return pool.filter(m => {
    if (query && !(m.text || '').toLowerCase().includes(query)) return false;
    if (emotionFilter) {
      const emotions = (m.metadata && m.metadata.emotions) || [];
      if (!emotions.includes(emotionFilter)) return false;
    }
    if (topicFilter) {
      const topic = m.metadata && m.metadata.topic;
      if (topic !== topicFilter) return false;
    }
    return true;
  });
}

function filterByEmotion(emotion) {
  filterEmotion.value = emotion;
  selectedDay = null;
  document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
  renderMemoryList();

  if (window.BrainViz) {
    window.BrainViz.deactivateAll();
    window.BrainViz.activateRegion('limbic');
    window.BrainViz.activateRegion('temporal');
  }
}

function filterByTopic(topic) {
  filterTopic.value = topic;
  selectedDay = null;
  document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
  renderMemoryList();

  if (window.BrainViz) {
    window.BrainViz.deactivateAll();
    window.BrainViz.activateRegion('frontal');
    window.BrainViz.activateRegion('temporal');
  }
}

// === Detail Modal ===
function openDetail(memory) {
  detailOverlay.classList.remove('hidden');

  if (window.BrainViz) {
    window.BrainViz.activateForMemory(memory);
  }

  const ts = memory._originalTimestamp || memory.timestamp || '';
  document.getElementById('detail-timestamp').textContent = ts.replace('T', ' ');

  const topic = memory.metadata && memory.metadata.topic;
  const topicEl = document.getElementById('detail-topic');
  if (topic) {
    topicEl.textContent = topic;
    topicEl.className = 'tag tag-topic';
    topicEl.style.display = '';
  } else {
    topicEl.style.display = 'none';
  }

  // Full text — strip exchange header line to avoid doubled prefixes
  const detailText = document.getElementById('detail-text');
  let fullText = memory.text || '';
  fullText = fullText.replace(/^exchange:\s*\[.*?\]\n?/i, '');
  detailText.textContent = fullText;

  // Metadata grid
  const metaGrid = document.getElementById('detail-meta-grid');
  metaGrid.innerHTML = '';
  const meta = memory.metadata || {};

  // Show all metadata keys
  const shownKeys = new Set(['emotions', 'related_memories', 'topic']); // skip these (shown elsewhere)
  const entries = [['id', memory.id], ['mse_loss', memory.mse_loss]];
  for (const [k, v] of Object.entries(meta)) {
    if (!shownKeys.has(k) && v !== undefined && v !== null) {
      entries.push([k, v]);
    }
  }
  for (const [key, val] of entries) {
    if (val === undefined || val === null) continue;
    const item = document.createElement('div');
    item.className = 'meta-item';
    const displayVal = typeof val === 'object' ? JSON.stringify(val) : String(val);
    item.innerHTML = `<span class="meta-key">${escapeHtml(key)}:</span><span class="meta-value">${escapeHtml(displayVal)}</span>`;
    metaGrid.appendChild(item);
  }

  // Emotions
  const emotionTags = document.getElementById('detail-emotion-tags');
  const emotions = (meta.emotions) || [];
  if (emotions.length > 0) {
    emotionTags.innerHTML = emotions.map(e =>
      `<span class="tag tag-emotion" data-emotion="${escapeHtml(e)}" style="cursor:pointer">${escapeHtml(e)}</span>`
    ).join('');
    document.getElementById('detail-emotions').style.display = '';

    emotionTags.querySelectorAll('.tag-emotion').forEach(tag => {
      tag.addEventListener('click', () => {
        closeDetail();
        filterByEmotion(tag.dataset.emotion);
      });
    });
  } else {
    emotionTags.innerHTML = '<span class="tag tag-notag">no emotions detected</span>';
    document.getElementById('detail-emotions').style.display = '';
  }

  // Related memories
  const relatedList = document.getElementById('detail-related-list');
  const relatedIds = (meta.related_memories) || [];
  if (relatedIds.length > 0) {
    relatedList.innerHTML = '';
    for (const rid of relatedIds) {
      const related = memoriesById[rid];
      const item = document.createElement('div');
      item.className = 'related-item';
      if (related) {
        const preview = escapeHtml((related.text || '').substring(0, 120));
        item.innerHTML = `<span class="related-id">#${rid}</span>${preview}`;
        item.addEventListener('click', () => {
          openDetail(related);
        });
      } else {
        item.innerHTML = `<span class="related-id">#${rid}</span><span style="color:var(--text-dim)">memory not loaded</span>`;
      }
      relatedList.appendChild(item);
    }
    document.getElementById('detail-related').style.display = '';
  } else {
    relatedList.innerHTML = '<span class="tag tag-notag">no linked memories</span>';
    document.getElementById('detail-related').style.display = '';
  }

  // Draw mini connection graph
  requestAnimationFrame(() => drawGraph(memory, emotions, relatedIds, topic));
}

function closeDetail() {
  detailOverlay.classList.add('hidden');
  if (window.BrainViz) {
    window.BrainViz.deactivateAll();
  }
}

// === Mini Connection Graph ===
function drawGraph(memory, emotions, relatedIds, topic) {
  const canvas = document.getElementById('graph-canvas');
  const container = canvas.parentElement;
  const ctx = canvas.getContext('2d');
  const w = container.clientWidth || 700;
  const h = 220;
  canvas.width = w * window.devicePixelRatio;
  canvas.height = h * window.devicePixelRatio;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  ctx.clearRect(0, 0, w, h);

  const centerX = w / 2;
  const centerY = h / 2;

  const nodes = [];
  // Center: this memory
  nodes.push({ x: centerX, y: centerY, label: `#${memory.id}`, color: '#00d4ff', r: 20 });

  // Emotion nodes (left arc)
  if (emotions.length > 0) {
    const spread = Math.min(emotions.length * 40, h - 40);
    const startY = centerY - spread / 2;
    emotions.forEach((e, i) => {
      const y = startY + (spread / Math.max(emotions.length - 1, 1)) * i;
      nodes.push({ x: centerX - 180, y: emotions.length === 1 ? centerY : y, label: e, color: '#ec4899', r: 13 });
    });
  }

  // Topic node (top)
  if (topic) {
    nodes.push({ x: centerX, y: 28, label: topic, color: '#a855f7', r: 15 });
  }

  // Related memory nodes (right arc)
  if (relatedIds.length > 0) {
    const maxShow = Math.min(relatedIds.length, 5);
    const spread = Math.min(maxShow * 36, h - 40);
    const startY = centerY - spread / 2;
    for (let i = 0; i < maxShow; i++) {
      const y = startY + (spread / Math.max(maxShow - 1, 1)) * i;
      nodes.push({ x: centerX + 180, y: maxShow === 1 ? centerY : y, label: `#${relatedIds[i]}`, color: '#22d3ee', r: 13 });
    }
  }

  // Source label (bottom)
  const source = memory.metadata && memory.metadata.source;
  if (source) {
    nodes.push({ x: centerX, y: h - 24, label: source.replace('session_ingest_v3_', ''), color: '#f59e0b', r: 12 });
  }

  // Draw connections
  for (let i = 1; i < nodes.length; i++) {
    const n = nodes[i];
    const c = nodes[0];

    // Glow line
    ctx.strokeStyle = n.color + '25';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const cpx = (c.x + n.x) / 2;
    const cpy1 = c.y;
    const cpy2 = n.y;
    ctx.moveTo(c.x, c.y);
    ctx.bezierCurveTo(cpx, cpy1, cpx, cpy2, n.x, n.y);
    ctx.stroke();

    // Fine line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.bezierCurveTo(cpx, cpy1, cpx, cpy2, n.x, n.y);
    ctx.stroke();
  }

  // Draw nodes
  for (const node of nodes) {
    // Outer glow
    const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r * 3);
    grad.addColorStop(0, node.color + '30');
    grad.addColorStop(1, node.color + '00');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r * 3, 0, Math.PI * 2);
    ctx.fill();

    // Circle
    ctx.fillStyle = node.color + '20';
    ctx.strokeStyle = node.color + 'aa';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Label
    ctx.fillStyle = '#d0d0e8';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(node.label, node.x, node.y + node.r + 5);
  }
}

// === Event Listeners ===
function setupEventListeners() {
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      const q = searchInput.value.trim();
      if (q) {
        selectedDay = null;
        document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
        if (window.BrainViz) window.BrainViz.activateForSearch(q);
      } else {
        if (window.BrainViz) window.BrainViz.deactivateAll();
        // Re-select latest day
        const days = Object.keys(dayGroups).sort().reverse();
        if (days.length > 0) selectDay(days[0]);
      }
      renderMemoryList();
    }, 250);
  });

  filterEmotion.addEventListener('change', () => {
    if (filterEmotion.value) {
      selectedDay = null;
      document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
    }
    renderMemoryList();
    if (filterEmotion.value && window.BrainViz) {
      window.BrainViz.deactivateAll();
      window.BrainViz.activateRegion('limbic');
    }
  });

  filterTopic.addEventListener('change', () => {
    if (filterTopic.value) {
      selectedDay = null;
      document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
    }
    renderMemoryList();
    if (filterTopic.value && window.BrainViz) {
      window.BrainViz.deactivateAll();
      window.BrainViz.activateRegion('frontal');
    }
  });

  detailClose.addEventListener('click', closeDetail);
  detailOverlay.addEventListener('click', (e) => {
    if (e.target === detailOverlay) closeDetail();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDetail();
  });
}

// === Boot ===
init();
