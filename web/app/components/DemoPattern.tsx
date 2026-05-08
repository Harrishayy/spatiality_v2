// Deterministic decorative thumbnail for demo cards.
// Seeded by scene_id so the same scene always gets the same pattern + colour.
// Six variants drawn from the sunset palette — picked to read as "data /
// blueprint" (rings, hatch, topo, iso grid) rather than decorative wallpaper.

const PALETTE = [
  "rgba(255, 107,  74, 0.62)", // coral
  "rgba(255, 157, 111, 0.62)", // apricot
  "rgba(255, 195, 122, 0.62)", // gold
  "rgba(255,  93, 143, 0.55)", // pink-coral
  "rgba(255, 179,  71, 0.62)", // amber
];

function hash(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function DemoPattern({ seed }: { seed: string }) {
  const h = hash(seed);
  const variant = h % 6;
  const stroke = PALETTE[(h >> 5) % PALETTE.length];
  const accent = PALETTE[(h >> 11) % PALETTE.length];
  const safe = seed.replace(/[^a-z0-9]/gi, "").slice(0, 16) || "x";
  const patId = `dp-${safe}-${variant}`;

  return (
    <svg
      className="lp-demo-pattern"
      viewBox="0 0 200 125"
      preserveAspectRatio="xMidYMid slice"
      role="presentation"
      aria-hidden="true"
    >
      <defs>{patternDefs(variant, patId, stroke)}</defs>
      {variant <= 1 || variant === 5 ? (
        <rect width="200" height="125" fill={`url(#${patId})`} />
      ) : (
        <PatternShape variant={variant} stroke={stroke} accent={accent} />
      )}
    </svg>
  );
}

function patternDefs(variant: number, id: string, stroke: string) {
  switch (variant) {
    case 0: // diagonal stripes
      return (
        <pattern
          id={id}
          width="14"
          height="14"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(35)"
        >
          <line x1="0" y1="0" x2="0" y2="14" stroke={stroke} strokeWidth="1" />
        </pattern>
      );
    case 1: // dot grid
      return (
        <pattern id={id} width="14" height="14" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="1.1" fill={stroke} />
        </pattern>
      );
    case 5: // isometric grid
      return (
        <pattern
          id={id}
          width="18"
          height="18"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <line x1="0" y1="0" x2="18" y2="0" stroke={stroke} strokeWidth="0.7" />
          <line x1="0" y1="0" x2="0" y2="18" stroke={stroke} strokeWidth="0.7" />
        </pattern>
      );
    default:
      return null;
  }
}

function PatternShape({
  variant,
  stroke,
  accent,
}: {
  variant: number;
  stroke: string;
  accent: string;
}) {
  switch (variant) {
    case 2: // concentric arcs from bottom-left
      return (
        <g fill="none" strokeWidth="1.1">
          {[40, 70, 100, 130, 160, 190, 220, 250].map((r, i) => (
            <circle
              key={r}
              cx="0"
              cy="125"
              r={r}
              stroke={i % 3 === 0 ? accent : stroke}
              opacity={0.55 + (i % 2) * 0.15}
            />
          ))}
        </g>
      );
    case 3: // cross-hatch
      return (
        <g strokeWidth="0.8">
          {Array.from({ length: 22 }).map((_, i) => {
            const x = i * 14 - 60;
            return (
              <g key={i}>
                <line x1={x} y1="0" x2={x + 130} y2="125" stroke={stroke} />
                <line
                  x1={x}
                  y1="125"
                  x2={x + 130}
                  y2="0"
                  stroke={accent}
                  opacity="0.45"
                />
              </g>
            );
          })}
        </g>
      );
    case 4: // topographic rings (offset ellipses)
      return (
        <g fill="none" strokeWidth="1">
          {Array.from({ length: 9 }).map((_, i) => (
            <ellipse
              key={i}
              cx="135"
              cy="58"
              rx={14 + i * 14}
              ry={9 + i * 9}
              stroke={i % 2 === 0 ? stroke : accent}
              opacity={0.7 - i * 0.05}
            />
          ))}
        </g>
      );
    default:
      return null;
  }
}
