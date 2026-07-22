/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: '#4F6EF6',
          light: '#7B93F8',
          dark: '#3354E8',
        },
      },
    },
  },
  plugins: [],
}
