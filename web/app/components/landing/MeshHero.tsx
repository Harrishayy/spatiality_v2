"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export function MeshHero() {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const hostMaybe = ref.current;
    if (!hostMaybe) return;
    const host: HTMLDivElement = hostMaybe;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x1a0e14, 0.06);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(6.2, 3.4, 6.6);
    camera.lookAt(0, 0.2, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);

    const N = 1400;
    const targets = new Float32Array(N * 3);
    const labels = new Uint8Array(N);

    const rand = (min: number, max: number) =>
      min + Math.random() * (max - min);
    const jitter = (v: number, j: number) => v + (Math.random() - 0.5) * j;

    let i = 0;
    for (; i < 380; i++) {
      targets[i * 3 + 0] = rand(-3, 3);
      targets[i * 3 + 1] = jitter(0, 0.04);
      targets[i * 3 + 2] = rand(-3, 3);
      labels[i] = 0;
    }
    for (; i < 600; i++) {
      targets[i * 3 + 0] = rand(-3, 3);
      targets[i * 3 + 1] = rand(0, 2.6);
      targets[i * 3 + 2] = jitter(-3, 0.05);
      labels[i] = 1;
    }
    for (; i < 780; i++) {
      targets[i * 3 + 0] = jitter(-3, 0.05);
      targets[i * 3 + 1] = rand(0, 2.6);
      targets[i * 3 + 2] = rand(-3, 3);
      labels[i] = 1;
    }
    for (; i < 1020; i++) {
      const cx = -1.2,
        cy = 0.45,
        cz = 1.0;
      const sx = 1.0,
        sy = 0.45,
        sz = 0.45;
      const face = Math.floor(Math.random() * 5);
      let x = cx + rand(-sx, sx);
      let y = cy + rand(-sy, sy);
      let z = cz + rand(-sz, sz);
      if (face === 0) y = cy + sy;
      else if (face === 1) x = cx - sx;
      else if (face === 2) x = cx + sx;
      else if (face === 3) z = cz - sz;
      else if (face === 4) z = cz + sz;
      targets[i * 3 + 0] = jitter(x, 0.03);
      targets[i * 3 + 1] = jitter(y, 0.03);
      targets[i * 3 + 2] = jitter(z, 0.03);
      labels[i] = 2;
    }
    for (; i < 1200; i++) {
      const cx = 1.4,
        cy = 0.7,
        cz = -0.6;
      const sx = 0.7,
        sy = 0.05,
        sz = 0.5;
      const face = Math.floor(Math.random() * 5);
      let x = cx + rand(-sx, sx);
      let y = cy + rand(-sy, sy);
      let z = cz + rand(-sz, sz);
      if (face === 0) y = cy + sy;
      else if (face === 1) x = cx - sx;
      else if (face === 2) x = cx + sx;
      else if (face === 3) z = cz - sz;
      else if (face === 4) z = cz + sz;
      targets[i * 3 + 0] = jitter(x, 0.025);
      targets[i * 3 + 1] = jitter(y, 0.025);
      targets[i * 3 + 2] = jitter(z, 0.025);
      labels[i] = 3;
    }
    const legs: [number, number, number][] = [
      [0.85, 0.35, -0.15],
      [0.85, 0.35, -1.05],
      [1.95, 0.35, -0.15],
      [1.95, 0.35, -1.05],
    ];
    for (let li = 0; li < 4; li++, i += 50) {
      for (let k = 0; k < 50; k++) {
        targets[(i + k) * 3 + 0] = jitter(legs[li][0], 0.03);
        targets[(i + k) * 3 + 1] = jitter(legs[li][1], 0.32);
        targets[(i + k) * 3 + 2] = jitter(legs[li][2], 0.03);
        labels[i + k] = 3;
      }
    }
    const used = i;

    const origins = new Float32Array(used * 3);
    for (let k = 0; k < used; k++) {
      origins[k * 3 + 0] = rand(-7, 7);
      origins[k * 3 + 1] = rand(-3, 5);
      origins[k * 3 + 2] = rand(-7, 7);
    }

    const positions = new Float32Array(used * 3);
    const colors = new Float32Array(used * 3);
    const palette: [number, number, number][] = [
      [0.95, 0.78, 0.62],
      [0.55, 0.3, 0.42],
      [1.0, 0.42, 0.29],
      [1.0, 0.62, 0.43],
      [1.0, 0.82, 0.55],
    ];
    for (let k = 0; k < used; k++) {
      const c = palette[labels[k]] || palette[0];
      colors[k * 3 + 0] = c[0];
      colors[k * 3 + 1] = c[1];
      colors[k * 3 + 2] = c[2];
    }

    const pointGeom = new THREE.BufferGeometry();
    pointGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    pointGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const pointMat = new THREE.PointsMaterial({
      size: 0.038,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(pointGeom, pointMat);
    scene.add(points);

    const edgeIndices: number[] = [];
    const groups: number[][] = [[], [], [], [], []];
    for (let k = 0; k < used; k++) groups[labels[k]].push(k);

    function pickTriangles(group: number[], count: number) {
      for (let t = 0; t < count; t++) {
        if (group.length < 3) break;
        const a = group[Math.floor(Math.random() * group.length)];
        const tries = 6;
        let b = -1,
          c = -1;
        let bd = 999,
          cd = 999;
        const ax = targets[a * 3],
          ay = targets[a * 3 + 1],
          az = targets[a * 3 + 2];
        for (let q = 0; q < tries; q++) {
          const cand = group[Math.floor(Math.random() * group.length)];
          if (cand === a) continue;
          const dx = targets[cand * 3] - ax;
          const dy = targets[cand * 3 + 1] - ay;
          const dz = targets[cand * 3 + 2] - az;
          const d = dx * dx + dy * dy + dz * dz;
          if (d < bd) {
            cd = bd;
            c = b;
            bd = d;
            b = cand;
          } else if (d < cd) {
            cd = d;
            c = cand;
          }
        }
        if (b >= 0 && c >= 0 && b !== c) {
          edgeIndices.push(a, b, b, c, c, a);
        }
      }
    }
    pickTriangles(groups[0], 90);
    pickTriangles(groups[1], 110);
    pickTriangles(groups[2], 120);
    pickTriangles(groups[3], 90);

    const edgePos = new Float32Array(edgeIndices.length * 3);
    const edgeCol = new Float32Array(edgeIndices.length * 3);
    for (let e = 0; e < edgeIndices.length; e++) {
      const idx = edgeIndices[e];
      edgePos[e * 3 + 0] = targets[idx * 3 + 0];
      edgePos[e * 3 + 1] = targets[idx * 3 + 1];
      edgePos[e * 3 + 2] = targets[idx * 3 + 2];
      const c = palette[labels[idx]] || palette[0];
      edgeCol[e * 3 + 0] = c[0];
      edgeCol[e * 3 + 1] = c[1];
      edgeCol[e * 3 + 2] = c[2];
    }
    const edgeGeom = new THREE.BufferGeometry();
    edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
    edgeGeom.setAttribute("color", new THREE.BufferAttribute(edgeCol, 3));
    const edgeMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0,
    });
    const edges = new THREE.LineSegments(edgeGeom, edgeMat);
    scene.add(edges);

    // Pin positions are seated on the actual top of each object's point
    // distribution so the marker dot lands on the geometry, not floating
    // above it. Label is offset +0.18 in y at projection time below.
    //   couch box: cy=0.45, sy=0.45 → top y=0.90
    //   table top slab: cy=0.70, sy=0.05 → top y=0.75
    //   floor slab: y≈0
    const pinSpec: { label: string; pos: [number, number, number] }[] = [
      { label: "couch", pos: [-1.2, 0.92, 1.0] },
      { label: "table", pos: [1.4, 0.78, -0.6] },
      { label: "floor", pos: [0.6, 0.02, 0.6] },
    ];
    // Build a canvas-based texture for each pin's label. The pill is drawn
    // once and used as a Sprite material — Sprites are billboards, so the
    // label always faces the camera and its world-space position is
    // automatically projected by THREE itself. No manual CSS sync.
    const dpr = Math.min(window.devicePixelRatio, 2);
    function makeLabelTexture(text: string): THREE.CanvasTexture {
      const W = 256, H = 64;
      const canvas = document.createElement("canvas");
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);
      // pill background
      ctx.fillStyle = "rgba(38,21,32,0.92)";
      ctx.strokeStyle = "rgba(255,157,111,0.55)";
      ctx.lineWidth = 1.5;
      const padX = 16, padY = 18, w = W - padX * 2, h = H - padY * 2, r = h / 2;
      ctx.beginPath();
      ctx.moveTo(padX + r, padY);
      ctx.lineTo(padX + w - r, padY);
      ctx.arcTo(padX + w, padY, padX + w, padY + r, r);
      ctx.lineTo(padX + w, padY + h - r);
      ctx.arcTo(padX + w, padY + h, padX + w - r, padY + h, r);
      ctx.lineTo(padX + r, padY + h);
      ctx.arcTo(padX, padY + h, padX, padY + h - r, r);
      ctx.lineTo(padX, padY + r);
      ctx.arcTo(padX, padY, padX + r, padY, r);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // text
      ctx.fillStyle = "#ffd29c";
      ctx.font = "600 16px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text.toUpperCase(), W / 2, H / 2);
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      return tex;
    }

    const pins = pinSpec.map((p) => {
      const g = new THREE.SphereGeometry(0.045, 12, 12);
      const m = new THREE.MeshBasicMaterial({
        color: 0xffb347,
        transparent: true,
        opacity: 0,
      });
      const s = new THREE.Mesh(g, m);
      s.position.set(p.pos[0], p.pos[1], p.pos[2]);
      scene.add(s);
      const hg = new THREE.RingGeometry(0.07, 0.1, 24);
      const hm = new THREE.MeshBasicMaterial({
        color: 0xff9d6f,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
      });
      const halo = new THREE.Mesh(hg, hm);
      halo.position.copy(s.position);
      scene.add(halo);

      // Label sprite — child of the pin sphere so its world position is
      // s.position + local offset, automatically. Always faces the camera.
      const tex = makeLabelTexture(p.label);
      const spriteMat = new THREE.SpriteMaterial({
        map: tex,
        transparent: true,
        opacity: 0,
        depthTest: false,
        depthWrite: false,
      });
      const sprite = new THREE.Sprite(spriteMat);
      // Hover ~22cm above the pin in world units. 256x64 canvas → 4:1 ratio.
      sprite.position.set(0, 0.22, 0);
      sprite.scale.set(0.7, 0.175, 1);
      // Render labels on top of all other geometry.
      sprite.renderOrder = 999;
      s.add(sprite);

      return { core: s, halo, label: p.label, pos: p.pos, sprite, spriteMat, tex };
    });

    // Object-anchor wireframes — without these the couch/table point clouds
    // get lost in the global edge mesh and the labels look like they're
    // floating in empty space. Each box matches the actual cluster extent
    // (centers/sizes pulled from the seed loops above) so the wireframe
    // visually wraps the very points it's labeling.
    type ObjectBox = {
      key: "couch" | "table";
      center: [number, number, number];
      size: [number, number, number];
      color: number;
    };
    const objectBoxes: ObjectBox[] = [
      // Couch: center (-1.2, 0.45, 1.0), half-extents (1.0, 0.45, 0.45) → full size
      { key: "couch", center: [-1.2, 0.45, 1.0], size: [2.0, 0.9, 0.9], color: 0xff5d8f },
      // Table: include legs in the bounding box so the whole "table object"
      // is wrapped (top slab + 4 legs spanning y∈[0, 0.75]).
      { key: "table", center: [1.4, 0.375, -0.6], size: [1.5, 0.75, 1.05], color: 0xffb347 },
    ];
    const objectWireframes = objectBoxes.map((b) => {
      const boxGeo = new THREE.BoxGeometry(b.size[0], b.size[1], b.size[2]);
      const edgeGeo = new THREE.EdgesGeometry(boxGeo);
      boxGeo.dispose();
      const mat = new THREE.LineBasicMaterial({
        color: b.color,
        transparent: true,
        opacity: 0,
      });
      const lines = new THREE.LineSegments(edgeGeo, mat);
      lines.position.set(b.center[0], b.center[1], b.center[2]);
      scene.add(lines);
      return { lines, mat, edgeGeo };
    });

    // Floor anchor disc — the floor is an infinite plane so a wireframe box
    // doesn't apply; lay a flat ring at the floor pin so the label has a
    // visible target on the ground.
    const floorRingGeo = new THREE.RingGeometry(0.32, 0.42, 48);
    const floorRingMat = new THREE.MeshBasicMaterial({
      color: 0xffd29c,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
    });
    const floorRing = new THREE.Mesh(floorRingGeo, floorRingMat);
    floorRing.position.set(0.6, 0.012, 0.6);
    floorRing.rotation.x = -Math.PI / 2;
    scene.add(floorRing);

    // (Labels are THREE Sprites parented to their pins — see makeLabelTexture
    // above. No HTML overlay needed; THREE handles the projection itself.)

    function fit() {
      const r = host.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / r.height || 1;
      camera.updateProjectionMatrix();
    }
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(host);

    const CYCLE_MS = 8000;
    const start = performance.now();
    let raf = 0;

    const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
    const smooth = (t: number) => t * t * (3 - 2 * t);

    function tick() {
      const now = performance.now();
      const elapsed = (now - start) % CYCLE_MS;
      const u = elapsed / CYCLE_MS;

      let aP = 0;
      if (u < 0.3) aP = easeOut(u / 0.3);
      else if (u < 0.95) aP = 1;
      else aP = 1 - smooth((u - 0.95) / 0.05);

      const posAttr = pointGeom.attributes.position as THREE.BufferAttribute;
      const arr = posAttr.array as Float32Array;
      for (let k = 0; k < used; k++) {
        const ox = origins[k * 3 + 0];
        const oy = origins[k * 3 + 1];
        const oz = origins[k * 3 + 2];
        const tx = targets[k * 3 + 0];
        const ty = targets[k * 3 + 1];
        const tz = targets[k * 3 + 2];
        const delay = (k % 12) / 60;
        const local = Math.max(0, Math.min(1, (u - delay) / 0.3));
        const a = u < 0.3 ? easeOut(local) : aP;
        arr[k * 3 + 0] = ox + (tx - ox) * a;
        arr[k * 3 + 1] = oy + (ty - oy) * a;
        arr[k * 3 + 2] = oz + (tz - oz) * a;
      }
      posAttr.needsUpdate = true;

      // Edges are an explanatory beat between points and labels — they fade
      // OUT before the labels reach full opacity so the bounding boxes
      // (drawn during pinOp) aren't competing with the global edge mesh.
      let eOp = 0;
      if (u >= 0.30 && u < 0.45) eOp = (u - 0.30) / 0.15;
      else if (u >= 0.45 && u < 0.55) eOp = 1;
      else if (u >= 0.55 && u < 0.72) eOp = 1 - (u - 0.55) / 0.17;
      else eOp = 0;
      edgeMat.opacity = Math.max(0, Math.min(0.7, eOp));

      let splat = 0;
      if (u >= 0.55 && u < 0.95) splat = (u - 0.55) / 0.4;
      pointMat.size = 0.038 + 0.022 * smooth(splat);
      pointMat.opacity = 0.9 + 0.1 * splat;

      // Pins ride a longer plateau so labels are readable, not a flash:
      //   0.42 → 0.58  fade in  (overlaps the tail of the edge pulse so
      //                          labels feel like they snap to the mesh)
      //   0.58 → 0.90  hold at full opacity  (~2.5s of stable read time)
      //   0.90 → 1.00  fade out together with the rest of the cycle
      let pinOp = 0;
      if (u >= 0.42 && u < 0.58) pinOp = smooth((u - 0.42) / 0.16);
      else if (u >= 0.58 && u < 0.9) pinOp = 1;
      else if (u >= 0.9) pinOp = 1 - (u - 0.9) / 0.1;
      pinOp = Math.max(0, Math.min(1, pinOp));
      for (let p = 0; p < pins.length; p++) {
        (pins[p].core.material as THREE.MeshBasicMaterial).opacity = pinOp;
        (pins[p].halo.material as THREE.MeshBasicMaterial).opacity = pinOp * 0.5;
        pins[p].halo.scale.setScalar(1 + 0.6 * Math.sin(now * 0.004 + p));
        pins[p].halo.lookAt(camera.position);
        // Sprite labels — opacity tracks pinOp; position is automatic
        // because each sprite is a child of its pin.
        pins[p].spriteMat.opacity = pinOp;
      }
      for (const w of objectWireframes) {
        w.mat.opacity = pinOp * 0.85;
      }
      floorRingMat.opacity = pinOp * 0.7;
      floorRing.scale.setScalar(1 + 0.18 * Math.sin(now * 0.003));

      const t = now * 0.00015;
      camera.position.x = Math.cos(t) * 7.0;
      camera.position.z = Math.sin(t) * 7.0;
      camera.position.y = 3.0 + Math.sin(t * 1.3) * 0.4;
      camera.lookAt(0, 0.4, 0);

      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      pointGeom.dispose();
      edgeGeom.dispose();
      pointMat.dispose();
      edgeMat.dispose();
      pins.forEach((p) => {
        p.core.geometry.dispose();
        (p.core.material as THREE.Material).dispose();
        p.halo.geometry.dispose();
        (p.halo.material as THREE.Material).dispose();
        p.spriteMat.dispose();
        p.tex.dispose();
      });
      objectWireframes.forEach((w) => {
        w.edgeGeo.dispose();
        w.mat.dispose();
      });
      floorRingGeo.dispose();
      floorRingMat.dispose();
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={ref}
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
      }}
    />
  );
}
