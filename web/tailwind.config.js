/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        orb: {
          gray: "#64748B",
          blue: "#3B82F6",
          green: "#22C55E",
          orange: "#F97316",
          purple: "#A855F7",
        }
      }
    },
  },
  plugins: [],
}
