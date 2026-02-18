import * as THREE from 'three';

// =============================================
// 3D Brain Visualizer - Three.js
// =============================================

const canvas = document.getElementById('brain-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0.5, 6);
camera.lookAt(0, 0, 0);

// === Brain Region Definitions ===
const REGIONS = {
  frontal:   { label: 'Frontal',   categories: ['reasoning', 'planning', 'logic', 'analysis', 'decision'],
               color: new THREE.Color(0x00d4ff), pos: new THREE.Vector3(0, 0.7, 0.8) },
  temporal:  { label: 'Temporal',  categories: ['language', 'memory recall', 'conversation', 'personal info', 'names'],
               color: new THREE.Color(0xa855f7), pos: new THREE.Vector3(-1.1, -0.1, 0.3) },
  temporalR: { label: 'Temporal R',categories: ['music', 'tone', 'sound'],
               color: new THREE.Color(0x8b5cf6), pos: new THREE.Vector3(1.1, -0.1, 0.3) },
  limbic:    { label: 'Limbic',    categories: ['emotions', 'feelings', 'love', 'fear', 'joy', 'sadness', 'anger', 'curiosity', 'happiness', 'anxiety', 'excitement', 'pride', 'affection', 'frustration', 'nostalgia', 'reflection'],
               color: new THREE.Color(0xec4899), pos: new THREE.Vector3(0, -0.2, 0) },
  occipital: { label: 'Occipital', categories: ['visual', 'creative', 'art', 'image', 'design'],
               color: new THREE.Color(0x22d3ee), pos: new THREE.Vector3(0, 0.2, -0.9) },
  parietal:  { label: 'Parietal',  categories: ['spatial', 'math', 'numbers', 'code', 'technical'],
               color: new THREE.Color(0xf59e0b), pos: new THREE.Vector3(0, 1.0, -0.1) },
};

// === Lighting ===
const ambientLight = new THREE.AmbientLight(0x223344, 0.6);
scene.add(ambientLight);

const pointLight1 = new THREE.PointLight(0x00d4ff, 1.5, 20);
pointLight1.position.set(3, 3, 5);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0xa855f7, 1.0, 20);
pointLight2.position.set(-3, -2, 3);
scene.add(pointLight2);

// === Brain Geometry ===
// Main brain shape: an elongated sphere with bumps
const brainGroup = new THREE.Group();
scene.add(brainGroup);

// Brain base mesh
const brainGeometry = new THREE.SphereGeometry(1.2, 64, 48);
// Deform to brain-like shape
const positions = brainGeometry.attributes.position;
for (let i = 0; i < positions.count; i++) {
  const x = positions.getX(i);
  const y = positions.getY(i);
  const z = positions.getZ(i);

  // Elongate front-to-back
  const newZ = z * 1.1;
  // Widen at sides
  const newX = x * (1.0 + 0.15 * Math.max(0, -y));
  // Flatten bottom slightly
  const newY = y < -0.3 ? y * 0.7 : y;

  // Add organic bumps (brain folds illusion)
  const bump = Math.sin(x * 8 + y * 5) * 0.03 + Math.sin(y * 12 + z * 7) * 0.02;

  positions.setXYZ(i, newX + bump, newY + bump * 0.5, newZ);
}
brainGeometry.computeVertexNormals();

// Brain material - translucent with wireframe effect
const brainMaterial = new THREE.MeshPhongMaterial({
  color: 0x1a1a3e,
  emissive: 0x0a0a1e,
  specular: 0x334466,
  shininess: 30,
  transparent: true,
  opacity: 0.25,
  wireframe: false,
  side: THREE.DoubleSide,
});

const brainMesh = new THREE.Mesh(brainGeometry, brainMaterial);
brainGroup.add(brainMesh);

// Wireframe overlay
const wireGeo = brainGeometry.clone();
const wireMat = new THREE.MeshBasicMaterial({
  color: 0x223355,
  wireframe: true,
  transparent: true,
  opacity: 0.08,
});
const wireMesh = new THREE.Mesh(wireGeo, wireMat);
brainGroup.add(wireMesh);

// Central fissure (the line between left/right hemispheres)
const fissureCurve = new THREE.CatmullRomCurve3([
  new THREE.Vector3(0, -1.0, -1.2),
  new THREE.Vector3(0, -0.5, -0.8),
  new THREE.Vector3(0, 0.2, 0),
  new THREE.Vector3(0, 0.8, 0.6),
  new THREE.Vector3(0, 0.5, 1.1),
]);
const fissureGeo = new THREE.TubeGeometry(fissureCurve, 40, 0.008, 8, false);
const fissureMat = new THREE.MeshBasicMaterial({ color: 0x556688, transparent: true, opacity: 0.3 });
const fissureMesh = new THREE.Mesh(fissureGeo, fissureMat);
brainGroup.add(fissureMesh);

// === Brain Regions (glowing spheres) ===
const regionMeshes = {};
const regionGlows = {};

for (const [key, region] of Object.entries(REGIONS)) {
  // Region sphere
  const geo = new THREE.SphereGeometry(0.18, 32, 24);
  const mat = new THREE.MeshPhongMaterial({
    color: region.color,
    emissive: region.color.clone().multiplyScalar(0.2),
    transparent: true,
    opacity: 0.35,
    shininess: 80,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(region.pos);
  brainGroup.add(mesh);
  regionMeshes[key] = mesh;

  // Glow sphere (larger, more transparent)
  const glowGeo = new THREE.SphereGeometry(0.3, 24, 18);
  const glowMat = new THREE.MeshBasicMaterial({
    color: region.color,
    transparent: true,
    opacity: 0.0,
    side: THREE.BackSide,
  });
  const glowMesh = new THREE.Mesh(glowGeo, glowMat);
  glowMesh.position.copy(region.pos);
  brainGroup.add(glowMesh);
  regionGlows[key] = glowMesh;
}

// === Neural Network Lines (subtle connections between regions) ===
const connectionMaterial = new THREE.LineBasicMaterial({
  color: 0x334466,
  transparent: true,
  opacity: 0.08,
});

const regionKeys = Object.keys(REGIONS);
for (let i = 0; i < regionKeys.length; i++) {
  for (let j = i + 1; j < regionKeys.length; j++) {
    const p1 = REGIONS[regionKeys[i]].pos;
    const p2 = REGIONS[regionKeys[j]].pos;
    const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    mid.y += 0.2;
    const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(20));
    const line = new THREE.Line(lineGeo, connectionMaterial.clone());
    brainGroup.add(line);
  }
}

// === Particle System (ambient floating particles) ===
const PARTICLE_COUNT = 300;
const particleGeometry = new THREE.BufferGeometry();
const particlePositions = new Float32Array(PARTICLE_COUNT * 3);
const particleSpeeds = new Float32Array(PARTICLE_COUNT);
const particleColors = new Float32Array(PARTICLE_COUNT * 3);

for (let i = 0; i < PARTICLE_COUNT; i++) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = 1.0 + Math.random() * 0.8;
  particlePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
  particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.85;
  particlePositions[i * 3 + 2] = r * Math.cos(phi) * 1.1;
  particleSpeeds[i] = 0.002 + Math.random() * 0.004;

  // Subtle color variation
  const hue = 0.55 + Math.random() * 0.15;
  const col = new THREE.Color().setHSL(hue, 0.6, 0.6);
  particleColors[i * 3] = col.r;
  particleColors[i * 3 + 1] = col.g;
  particleColors[i * 3 + 2] = col.b;
}

particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
particleGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

const particleMaterial = new THREE.PointsMaterial({
  size: 0.02,
  vertexColors: true,
  transparent: true,
  opacity: 0.5,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const particles = new THREE.Points(particleGeometry, particleMaterial);
brainGroup.add(particles);

// === Activation Particles (flow between regions when memory selected) ===
const FLOW_COUNT = 80;
const flowPositions = new Float32Array(FLOW_COUNT * 3);
const flowColors = new Float32Array(FLOW_COUNT * 3);
const flowGeometry = new THREE.BufferGeometry();
flowGeometry.setAttribute('position', new THREE.BufferAttribute(flowPositions, 3));
flowGeometry.setAttribute('color', new THREE.BufferAttribute(flowColors, 3));

const flowMaterial = new THREE.PointsMaterial({
  size: 0.04,
  vertexColors: true,
  transparent: true,
  opacity: 0.8,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const flowParticles = new THREE.Points(flowGeometry, flowMaterial);
brainGroup.add(flowParticles);

// Flow state
const flowState = {
  active: false,
  sourceRegion: null,
  targetRegions: [],
  progress: new Float32Array(FLOW_COUNT),
  speeds: new Float32Array(FLOW_COUNT),
  targets: new Int32Array(FLOW_COUNT),
};

// === State ===
const activeRegions = new Set();
let time = 0;

// === Public API ===
window.BrainViz = {
  REGIONS,

  activateRegion(key) {
    activeRegions.add(key);
  },

  deactivateAll() {
    activeRegions.clear();
    flowState.active = false;
  },

  activateForMemory(memory) {
    activeRegions.clear();
    if (!memory) return;

    const text = (memory.text || '').toLowerCase();
    const topic = ((memory.metadata && memory.metadata.topic) || '').toLowerCase();
    const emotions = (memory.metadata && memory.metadata.emotions) || [];

    for (const [key, region] of Object.entries(REGIONS)) {
      for (const cat of region.categories) {
        if (text.includes(cat) || topic.includes(cat) || emotions.some(e => cat.includes(e.toLowerCase()))) {
          activeRegions.add(key);
          break;
        }
      }
    }

    // If emotions present, always activate limbic
    if (emotions.length > 0) activeRegions.add('limbic');
    // Always activate temporal for memory recall
    activeRegions.add('temporal');
    // If nothing specific matched, also light frontal
    if (activeRegions.size <= 1) activeRegions.add('frontal');

    // Start flow particles
    startFlow();
  },

  activateForSearch(query) {
    activeRegions.clear();
    if (!query) return;
    const q = query.toLowerCase();
    for (const [key, region] of Object.entries(REGIONS)) {
      for (const cat of region.categories) {
        if (q.includes(cat)) {
          activeRegions.add(key);
        }
      }
    }
    if (activeRegions.size === 0) {
      activeRegions.add('frontal');
      activeRegions.add('temporal');
    }
    startFlow();
  },

  pulseRegion(key) {
    if (regionMeshes[key]) {
      regionMeshes[key].scale.set(1.5, 1.5, 1.5);
    }
  }
};

function startFlow() {
  flowState.active = true;
  const activeArr = Array.from(activeRegions);
  if (activeArr.length < 1) return;

  for (let i = 0; i < FLOW_COUNT; i++) {
    flowState.progress[i] = Math.random();
    flowState.speeds[i] = 0.005 + Math.random() * 0.01;
    flowState.targets[i] = Math.floor(Math.random() * activeArr.length);

    const src = REGIONS[activeArr[0]];
    const col = src.color;
    flowColors[i * 3] = col.r;
    flowColors[i * 3 + 1] = col.g;
    flowColors[i * 3 + 2] = col.b;
  }
  flowGeometry.attributes.color.needsUpdate = true;
}

// === Animation Loop ===
function animate() {
  requestAnimationFrame(animate);
  time += 0.01;

  // Slow auto-rotation
  brainGroup.rotation.y = Math.sin(time * 0.3) * 0.3;
  brainGroup.rotation.x = Math.sin(time * 0.2) * 0.08 + 0.1;

  // Animate ambient particles
  const pPos = particleGeometry.attributes.position.array;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const idx = i * 3;
    const speed = particleSpeeds[i];
    // Orbit around brain
    const x = pPos[idx];
    const z = pPos[idx + 2];
    const angle = Math.atan2(z, x) + speed;
    const r = Math.sqrt(x * x + z * z);
    pPos[idx] = r * Math.cos(angle);
    pPos[idx + 2] = r * Math.sin(angle);
    pPos[idx + 1] += Math.sin(time + i) * 0.001;
  }
  particleGeometry.attributes.position.needsUpdate = true;

  // Animate region glow
  for (const [key, glowMesh] of Object.entries(regionGlows)) {
    const mesh = regionMeshes[key];
    const isActive = activeRegions.has(key);
    const targetOpacity = isActive ? 0.25 + Math.sin(time * 3 + key.length) * 0.1 : 0.0;
    const targetEmissive = isActive ? 0.6 : 0.2;
    const targetScale = isActive ? 1.2 + Math.sin(time * 2) * 0.1 : 1.0;

    glowMesh.material.opacity += (targetOpacity - glowMesh.material.opacity) * 0.08;
    mesh.material.emissive.copy(REGIONS[key].color).multiplyScalar(
      mesh.material.emissive.r + (targetEmissive * REGIONS[key].color.r - mesh.material.emissive.r) * 0.05
    );
    mesh.material.opacity += ((isActive ? 0.7 : 0.35) - mesh.material.opacity) * 0.05;

    // Scale animation
    const s = mesh.scale.x + (targetScale - mesh.scale.x) * 0.05;
    mesh.scale.set(s, s, s);
    const gs = s * 1.6;
    glowMesh.scale.set(gs, gs, gs);
  }

  // Animate flow particles
  if (flowState.active) {
    const activeArr = Array.from(activeRegions);
    if (activeArr.length > 0) {
      const fPos = flowGeometry.attributes.position.array;
      const center = new THREE.Vector3(0, 0, 0);

      for (let i = 0; i < FLOW_COUNT; i++) {
        flowState.progress[i] += flowState.speeds[i];
        if (flowState.progress[i] > 1.0) {
          flowState.progress[i] = 0;
          flowState.targets[i] = Math.floor(Math.random() * activeArr.length);
        }

        const targetKey = activeArr[flowState.targets[i] % activeArr.length];
        const target = REGIONS[targetKey].pos;
        const t = flowState.progress[i];

        // Curved path from center to target
        const mid = new THREE.Vector3().addVectors(center, target).multiplyScalar(0.5);
        mid.y += 0.5 * Math.sin(t * Math.PI);

        const p = new THREE.Vector3();
        p.x = (1 - t) * (1 - t) * center.x + 2 * (1 - t) * t * mid.x + t * t * target.x;
        p.y = (1 - t) * (1 - t) * center.y + 2 * (1 - t) * t * mid.y + t * t * target.y;
        p.z = (1 - t) * (1 - t) * center.z + 2 * (1 - t) * t * mid.z + t * t * target.z;

        fPos[i * 3] = p.x;
        fPos[i * 3 + 1] = p.y;
        fPos[i * 3 + 2] = p.z;
      }
      flowGeometry.attributes.position.needsUpdate = true;
      flowMaterial.opacity = 0.6 + Math.sin(time * 4) * 0.2;
    }
  } else {
    flowMaterial.opacity *= 0.95;
  }

  // Subtle breathing of the brain
  const breathe = 1.0 + Math.sin(time * 0.8) * 0.01;
  brainMesh.scale.set(breathe, breathe, breathe);

  renderer.render(scene, camera);
}

animate();

// === Resize Handler ===
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

export { REGIONS };
