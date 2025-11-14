// postcss.config.js (New Way for Tailwind v4)
export default {
  plugins: {
    '@tailwindcss/postcss': {}, // <-- This is the correct plugin
    autoprefixer: {},
  },
}