import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import './index.css';

export default function App() {
  const mountRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
const handLandmarkerRef = useRef<HandLandmarker | null>(null);
const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
const controlsRef = useRef<OrbitControls | null>(null);
const engineRef = useRef<THREE.Group | null>(null);
const previousRightHandPositionRef = useRef<{ x: number; y: number } | null>(null);
const previousTwoHandDistanceRef = useRef<number | null>(null);
const rightHandStateRef = useRef<'unknown' | 'open' | 'closed'>('unknown');
const rightHandMovedRef = useRef(false);
const autoRotationRef = useRef(false);
const streamRef = useRef<MediaStream | null>(null);
const trackingFrameRef = useRef<number | null>(null);
const [cameraActive, setCameraActive] = useState(false);
const startCamera = async () => {
  try {
    setCameraStatus('Loading hand tracking model...');

    const vision = await FilesetResolver.forVisionTasks(
     '/mediapipe/wasm'
    );

    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });

    handLandmarkerRef.current = landmarker;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    });

    streamRef.current = stream;

    const video = videoRef.current;
    if (!video) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      landmarker.close();
      handLandmarkerRef.current = null;
      setCameraStatus('Camera element not found');
      return;
    }

    video.srcObject = stream;
    await video.play();

    setCameraActive(true);
    setCameraStatus('Camera active — show your hand');

    const detectHands = () => {
      const video = videoRef.current;
      const landmarker = handLandmarkerRef.current;

      if (!video || !landmarker || video.readyState < 2) {
        trackingFrameRef.current = requestAnimationFrame(detectHands);
        return;
      }

      const result = landmarker.detectForVideo(video, performance.now());
      const engine = engineRef.current;
      const camera = cameraRef.current;
      const controls = controlsRef.current;

      const distanceFromWrist = (landmarks: typeof result.landmarks[number], index: number) => {
        const wrist = landmarks[0];
        const point = landmarks[index];
        return Math.hypot(point.x - wrist.x, point.y - wrist.y);
      };
      const isOpenPalm = (landmarks: typeof result.landmarks[number]) => {
        const pairs: [number, number][] = [[8, 6], [12, 10], [16, 14], [20, 18]];
        const extended = pairs.filter(([tip, pip]) =>
          distanceFromWrist(landmarks, tip) > distanceFromWrist(landmarks, pip) * 1.12
        ).length;
        return extended >= 3;
      };

      // Raw webcam frames are not mirrored, so MediaPipe's selfie-oriented handedness labels are swapped.
      const rightIndex = result.handednesses.findIndex(
        (items) => items[0]?.categoryName === 'Left'
      );
      const leftIndex = result.handednesses.findIndex(
        (items) => items[0]?.categoryName === 'Right'
      );
      // Preserve single-hand control even if handedness classification is reversed or missing.
      const primaryHandIndex = rightIndex >= 0 ? rightIndex : (result.landmarks.length === 1 ? 0 : -1);
      const rightLandmarks = primaryHandIndex >= 0 ? result.landmarks[primaryHandIndex] : undefined;
      const resolvedLeftIndex = leftIndex >= 0 && leftIndex !== primaryHandIndex ? leftIndex : -1;
      const leftLandmarks = resolvedLeftIndex >= 0 ? result.landmarks[resolvedLeftIndex] : undefined;
      const rightIsOpen = rightLandmarks ? isOpenPalm(rightLandmarks) : false;
      const leftIsOpen = leftLandmarks ? isOpenPalm(leftLandmarks) : false;
      const leftIsClosed = Boolean(leftLandmarks && !leftIsOpen);
      const bothHandsOpen = Boolean(rightLandmarks && leftLandmarks && rightIsOpen && leftIsOpen);

      // Two open hands take priority: their separation controls zoom without rotating the model.
      if (bothHandsOpen && camera && controls) {
        if (previousTwoHandDistanceRef.current === null) {
          autoRotationRef.current = false;
          rightHandStateRef.current = 'open';
          rightHandMovedRef.current = false;
        }
        const rightWrist = rightLandmarks![0];
        const leftWrist = leftLandmarks![0];
        const handDistance = Math.hypot(rightWrist.x - leftWrist.x, rightWrist.y - leftWrist.y);
        const previousDistance = previousTwoHandDistanceRef.current;
        if (previousDistance !== null) {
          const delta = handDistance - previousDistance;
          if (Math.abs(delta) > 0.002) {
            const offset = camera.position.clone().sub(controls.target);
            const currentDistance = offset.length();
            const nextDistance = THREE.MathUtils.clamp(currentDistance * Math.exp(-delta * 2.2), 3, 18);
            camera.position.copy(controls.target).add(offset.normalize().multiplyScalar(nextDistance));
            controls.update();
          }
        }
        previousTwoHandDistanceRef.current = handDistance;
        previousRightHandPositionRef.current = null;
      } else {
        previousTwoHandDistanceRef.current = null;

        // Closing the left hand enables normal right-hand control after zooming;
        // it must not trigger automatic rotation.
        if (leftIsClosed) autoRotationRef.current = false;

        if (rightLandmarks && engine) {
          const wrist = rightLandmarks[0];
          const previous = previousRightHandPositionRef.current;
          if (rightIsOpen) {
            // Reopening the right hand stops automatic rotation; a new spin requires fresh hand movement.
            if (rightHandStateRef.current !== 'open') rightHandMovedRef.current = false;
            if (!leftIsClosed) autoRotationRef.current = false;
            rightHandStateRef.current = 'open';
            if (previous) {
              const deltaX = wrist.x - previous.x;
              const deltaY = wrist.y - previous.y;
              if (Math.abs(deltaX) + Math.abs(deltaY) > 0.0015) rightHandMovedRef.current = true;
              if (!bothHandsOpen) {
                engine.rotation.y += deltaX * 6;
                engine.rotation.x += deltaY * 4;
              }
            }
            previousRightHandPositionRef.current = { x: wrist.x, y: wrist.y };
          } else {
            // Start spinning only on an open-to-closed transition after the user has moved the model.
            if (rightHandStateRef.current === 'open' && rightHandMovedRef.current) {
              autoRotationRef.current = true;
            }
            rightHandStateRef.current = 'closed';
            previousRightHandPositionRef.current = null;
          }
        } else {
          previousRightHandPositionRef.current = null;
          previousTwoHandDistanceRef.current = null;
          // Do not change autoRotationRef here: losing tracking should not unexpectedly stop a spin.
        }
      }

      setCameraStatus(
        bothHandsOpen
          ? 'Both hands open — move apart to zoom'
          : leftIsClosed && autoRotationRef.current
            ? 'Left hand closed — automatic 3D rotation'
            : rightLandmarks
            ? rightIsOpen
              ? 'Right hand open — rotate engine'
              : autoRotationRef.current
                ? 'Right hand closed — automatic 3D rotation'
                : 'Right hand closed — move/open hand to rotate'
            : result.landmarks.length > 0
              ? `Hand detected: ${result.landmarks.length}`
              : 'Camera active — no hand detected'
      );

      trackingFrameRef.current = requestAnimationFrame(detectHands);
    };

    trackingFrameRef.current = requestAnimationFrame(detectHands);
  } catch (error) {
    console.error('Camera/hand tracking error:', error);
    setCameraStatus(
      error instanceof Error
        ? `Error: ${error.message}`
        : 'Could not start camera'
    );
  }
};
  const [cameraStatus, setCameraStatus] = useState('Gesture controls: not connected');
  const [resetView, setResetView] = useState<(() => void) | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#101318');
    scene.fog = new THREE.Fog('#101318', 13, 30);

    const camera = new THREE.PerspectiveCamera(
  38,
  mount.clientWidth / mount.clientHeight,
  0.1,
  100
);
cameraRef.current = camera;

camera.position.set(7, 5, 9);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: 'default',
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 3;
    controls.maxDistance = 18;
    controls.target.set(0, 0.3, 0);
    controls.update();

    scene.add(new THREE.HemisphereLight('#dce9ff', '#38302a', 2));

    const addLight = (
      color: THREE.ColorRepresentation,
      intensity: number,
      position: [number, number, number]
    ) => {
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(...position);
      light.castShadow = true;
      scene.add(light);
    };

    addLight('#ffffff', 3.5, [5, 8, 6]);
    addLight('#8bbcff', 2.4, [-6, 3, 2]);
    addLight('#ffb56b', 2, [1, 4, -7]);

    const metal = new THREE.MeshStandardMaterial({
      color: '#8d969e',
      metalness: 0.86,
      roughness: 0.28,
    });

    const darkMetal = new THREE.MeshStandardMaterial({
      color: '#252b32',
      metalness: 0.8,
      roughness: 0.35,
    });

    const alloy = new THREE.MeshStandardMaterial({
      color: '#c0c6cb',
      metalness: 0.9,
      roughness: 0.22,
    });

    const black = new THREE.MeshStandardMaterial({
      color: '#171b20',
      metalness: 0.48,
      roughness: 0.42,
    });

    const copper = new THREE.MeshStandardMaterial({
      color: '#b66b39',
      metalness: 0.78,
      roughness: 0.3,
    });

    const rubber = new THREE.MeshStandardMaterial({
      color: '#191b1e',
      metalness: 0.05,
      roughness: 0.8,
    });

    const engine = new THREE.Group();
    engineRef.current = engine;
    scene.add(engine);

    function part(
      geometry: THREE.BufferGeometry,
      material: THREE.Material,
      position: [number, number, number],
      rotation: [number, number, number] = [0, 0, 0],
      parent: THREE.Group = engine
    ) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...position);
      mesh.rotation.set(...rotation);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    }

    function box(
      size: [number, number, number],
      material: THREE.Material,
      position: [number, number, number],
      rotation: [number, number, number] = [0, 0, 0]
    ) {
      return part(new THREE.BoxGeometry(...size), material, position, rotation);
    }

    function cylinder(
      radiusTop: number,
      radiusBottom: number,
      height: number,
      material: THREE.Material,
      position: [number, number, number],
      axis: 'x' | 'y' | 'z' = 'y',
      segments = 24
    ) {
      const rotation: [number, number, number] =
        axis === 'x' ? [0, 0, Math.PI / 2]
        : axis === 'z' ? [Math.PI / 2, 0, 0]
        : [0, 0, 0];

      return part(
        new THREE.CylinderGeometry(radiusTop, radiusBottom, height, segments),
        material,
        position,
        rotation
      );
    }

    function torus(
      radius: number,
      tube: number,
      material: THREE.Material,
      position: [number, number, number],
      rotation: [number, number, number] = [0, 0, 0]
    ) {
      return part(
        new THREE.TorusGeometry(radius, tube, 10, 36),
        material,
        position,
        rotation
      );
    }

    // Main crankcase and lower oil pan
    box([3.8, 0.65, 2.15], darkMetal, [0, -0.35, 0]);
    box([3.35, 0.3, 1.8], metal, [0, -0.78, 0]);
    box([3.0, 0.12, 1.55], black, [0, -0.97, 0]);

    // Engine block
    box([3.35, 1.5, 1.85], metal, [0, 0.65, 0]);

    // Four cylinder bores and sleeves
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * 0.79;

      cylinder(0.32, 0.35, 0.12, darkMetal, [x, 1.43, 0], 'y', 32);
      cylinder(0.255, 0.255, 0.08, alloy, [x, 1.50, 0], 'y', 32);
      cylinder(0.18, 0.18, 0.08, black, [x, 1.55, 0], 'y', 32);

      // Spark plugs
      cylinder(0.075, 0.075, 0.32, alloy, [x, 1.72, 0.27]);
      cylinder(0.11, 0.11, 0.08, darkMetal, [x, 1.88, 0.27]);

      // Head bolts
      for (const dx of [-0.24, 0.24]) {
        for (const dz of [-0.5, 0.5]) {
          cylinder(0.055, 0.055, 0.09, alloy, [x + dx, 1.43, dz]);
        }
      }
    }

    // Cylinder head and valve cover
    box([3.55, 0.26, 1.95], alloy, [0, 1.62, 0]);
    box([3.15, 0.3, 1.42], darkMetal, [0, 1.9, 0]);
    box([2.8, 0.09, 1.1], metal, [0, 2.08, 0]);
    box([2.3, 0.045, 0.08], copper, [0, 2.11, 0]);

    // Front timing cover and crank pulley
    box([0.48, 1.35, 1.6], alloy, [1.85, 0.4, 0]);
    cylinder(0.55, 0.55, 0.32, darkMetal, [2.15, -0.25, 0], 'x', 40);
    cylinder(0.43, 0.43, 0.36, black, [2.34, -0.25, 0], 'x', 40);
    cylinder(0.19, 0.19, 0.4, alloy, [2.5, -0.25, 0], 'x', 32);
    torus(0.48, 0.035, rubber, [2.34, -0.25, 0], [0, 0, Math.PI / 2]);

    // Exhaust manifold: four pipes leading into a collector
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * 0.65;

      cylinder(0.105, 0.105, 0.62, copper, [x, 0.35, 1.08], 'y', 16);

      const bend = part(
        new THREE.TorusGeometry(0.24, 0.095, 12, 28, Math.PI),
        copper,
        [x + 0.1, 0.03, 1.2],
        [Math.PI / 2, 0, Math.PI / 2]
      );
      bend.scale.set(1, 1, 1);

      cylinder(0.1, 0.13, 0.9, copper, [x + 0.15, -0.02, 1.42], 'x', 16);
    }

    // Intake manifold and throttle body
    box([2.65, 0.2, 0.42], darkMetal, [0, 0.95, -1.08]);
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * 0.62;
      cylinder(0.12, 0.16, 0.52, alloy, [x, 0.72, -1.12], 'y', 20);
    }
    cylinder(0.29, 0.29, 0.42, darkMetal, [1.6, 0.98, -1.1], 'x', 32);
    cylinder(0.2, 0.2, 0.45, alloy, [1.85, 0.98, -1.1], 'x', 32);

    // Visible side bolts and fasteners
    for (let i = 0; i < 5; i++) {
      const x = -1.35 + i * 0.67;
      for (const y of [0.1, 0.75]) {
        cylinder(0.065, 0.065, 0.1, alloy, [x, y, 0.96], 'z', 6);
      }
    }

    // Alternator assembly
    cylinder(0.39, 0.39, 0.46, alloy, [-1.65, -0.1, 1.2], 'x', 32);
    cylinder(0.31, 0.31, 0.5, darkMetal, [-1.9, -0.1, 1.2], 'x', 32);
    torus(0.31, 0.04, alloy, [-1.95, -0.1, 1.2], [0, Math.PI / 2, 0]);

    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      cylinder(
        0.025, 0.025, 0.46, metal,
        [-1.9, -0.1 + Math.cos(angle) * 0.25, 1.2 + Math.sin(angle) * 0.25],
        'x', 8
      );
    }

    // Base and studio floor
    box([4.8, 0.14, 3.5], black, [0, -1.18, 0]);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({
        color: '#171b21',
        metalness: 0.35,
        roughness: 0.65,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.27;
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(30, 60, '#394655', '#242c36');
    grid.position.y = -1.265;
    scene.add(grid);

    const initialPosition = new THREE.Vector3(7, 5, 9);
    setResetView(() => () => {
      camera.position.copy(initialPosition);
      controls.target.set(0, 0.3, 0);
      controls.update();
    });

    let frameId = 0;
    let disposed = false;

    const animate = () => {
      if (disposed) return;
      frameId = requestAnimationFrame(animate);
      // Apply the spin continuously; the gesture detector only toggles this state.
      if (autoRotationRef.current && engineRef.current) {
        engineRef.current.rotation.y += 0.012;
        engineRef.current.rotation.x += 0.004;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resizeObserver = new ResizeObserver(() => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    });
    resizeObserver.observe(mount);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      if (trackingFrameRef.current !== null) {
        cancelAnimationFrame(trackingFrameRef.current);
        trackingFrameRef.current = null;
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      previousRightHandPositionRef.current = null;
      previousTwoHandDistanceRef.current = null;
      rightHandStateRef.current = 'unknown';
      rightHandMovedRef.current = false;
      autoRotationRef.current = false;
      cameraRef.current = null;
      controlsRef.current = null;
      engineRef.current = null;
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          if (Array.isArray(object.material)) {
            object.material.forEach((material) => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
      renderer.domElement.remove();
    };
  }, []);

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">EC</div>
          <div>
            <h1>ENGINE<span>CTRL</span></h1>
            <p>INTERACTIVE ENGINE LABORATORY</p>
          </div>
        </div>
        <div className="live-indicator">
          <span className="status-dot" />
          3D VIEWER ONLINE
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="eyebrow">MODEL 001 / POWERTRAIN</div>
          <h2>Engine<br /><span>Explorer</span></h2>
          <p className="description">
            Inspect a detailed 3D engine assembly. Rotate and zoom to explore
            its components from every angle.
          </p>

          <div className="divider" />

          <div className="section-label">VIEW CONTROLS</div>
          <div className="control-row">
            <span><kbd>DRAG</kbd> Rotate view</span>
          </div>
          <div className="control-row">
            <span><kbd>SCROLL</kbd> Zoom in / out</span>
          </div>
          <div className="control-row">
            <span><kbd>RIGHT HAND</kbd> Open: rotate · Close: auto-spin</span>
          </div>
          <div className="control-row">
            <span><kbd>BOTH HANDS</kbd> Open: apart to zoom in</span>
          </div>
          <div className="control-row">
            <span><kbd>RIGHT DRAG</kbd> Pan view</span>
          </div>

          <button className="reset-button" onClick={() => resetView?.()}>
            <span>↺</span> Reset camera
          </button>

          <div className="divider" />

          <div className="section-label">HAND TRACKING</div>
          <div className="camera-status">
            <span className="status-dot muted" />
            <span>{cameraStatus}</span>
          </div>
          <div className="hint">
            <video
  ref={videoRef}
  autoPlay
  playsInline
  muted
  style={{ width: '100%', borderRadius: '8px', marginTop: '10px' }}
/>

<button onClick={startCamera} disabled={cameraActive}>
  {cameraActive ? 'Camera Started' : 'Start Camera'}
</button>

<span>{cameraStatus}</span>
          </div>
        </aside>

        <section className="viewer-panel">
          <div className="viewer-heading">
            <div>
              <span className="viewer-kicker">3D WORKSPACE</span>
              <h3>Mechanical assembly</h3>
            </div>
            <span className="model-tag">ENGINE / 3D</span>
          </div>

          <div className="viewport" ref={mountRef}>
            <div className="viewport-badge">INTERACTIVE MODEL</div>
            <div className="viewport-help">HAND GESTURES · MOUSE DRAG · SCROLL TO ZOOM</div>
          </div>

          <footer className="viewer-footer">
            <span>RENDER ENGINE <strong>THREE.JS</strong></span>
            <span>INPUT <strong>MOUSE / TRACKPAD</strong></span>
          </footer>
        </section>
      </section>
    </main>
  );
}