/** @type {import('tailwindcss').Config} */
export default {
  // Scope the JIT scan to files that actually emit JSX. SplatViewer.tsx is
  // the heavy one; if it lived in a non-jsx subdir we'd exclude it. As-is,
  // we still need it scanned for its 39 className attrs, but excluding the
  // PLY parser body would be ideal — if SplatViewer is split later, narrow
  // this glob to ./app/{components,scenes,hooks}/**/*.tsx and ./app/*.tsx.
  content: ["./app/**/*.tsx"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#1a0e14",
          900: "#261520",
          800: "#36202c",
          700: "#4d2f3a",
          600: "#6d4651",
          500: "#94656a",
          400: "#c08a83",
          300: "#e6b9a3",
          200: "#f4dcc6",
          100: "#fdeede",
        },
        accent: {
          500: "#ff6b4a",
          400: "#ff9d6f",
          300: "#ffd29c",
        },
        hueMagenta: "#ff5d8f",
        hueAmber: "#ffb347",
        hueViolet: "#8b5fa8",
        emerald: "#4ec9b0",
      },
      fontFamily: {
        sans: [
          "var(--font-inter)",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        serif: [
          "var(--font-fraunces)",
          "Fraunces",
          "ui-serif",
          "Georgia",
          "serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      keyframes: {
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
        slide: {
          from: { transform: "translateY(8px)", opacity: "0" },
          to: { transform: "translateY(0)", opacity: "1" },
        },
      },
      animation: {
        "slide-in": "slide 200ms ease-out",
      },
    },
  },
  plugins: [],
};
