const { fontFamily } = require("tailwindcss/defaultTheme")

/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  safelist: [
    'border-slate-200',
    'dark:border-slate-800',
    'border-zinc-200',
    'dark:border-zinc-800',
    'text-emerald-600',
    'dark:text-emerald-400',
    'text-red-600',
    'dark:text-red-400',
    'bg-background',
    'text-foreground',
    'bg-card',
    'text-card-foreground',
    'bg-popover',
    'text-popover-foreground',
    'bg-primary',
    'text-primary-foreground',
    'bg-secondary',
    'text-secondary-foreground',
    'bg-muted',
    'text-muted-foreground',
    'bg-accent',
    'text-accent-foreground',
    'ring-offset-background',
    'border-input',
    'ring-ring',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        profit: {
          DEFAULT: "hsl(var(--profit))",
          light: "hsl(var(--profit-light))",
          dark: "hsl(var(--profit-dark))",
        },
        loss: {
          DEFAULT: "hsl(var(--loss))",
          light: "hsl(var(--loss-light))",
          dark: "hsl(var(--loss-dark))",
        },
        slate: {
          200: '#e2e8f0',
          800: '#1e293b',
        },
        zinc: {
          200: '#e4e4e7',
          800: '#27272a',
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", ...fontFamily.sans],
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
        "fade-in": {
          from: { opacity: 0 },
          to: { opacity: 1 },
        },
        "fade-out": {
          from: { opacity: 1 },
          to: { opacity: 0 },
        },
        "slide-in-from-right": {
          from: { transform: "translateX(100%)" },
          to: { transform: "translateX(0)" },
        },
        "slide-out-to-right": {
          from: { transform: "translateX(0)" },
          to: { transform: "translateX(100%)" },
        },
        "price-flash-up": {
          "0%, 100%": { backgroundColor: "transparent" },
          "50%": { backgroundColor: "hsla(var(--profit), 0.2)" },
        },
        "price-flash-down": {
          "0%, 100%": { backgroundColor: "transparent" },
          "50%": { backgroundColor: "hsla(var(--loss), 0.2)" },
        },
        "pulse-buy": {
          "0%": { boxShadow: "0 0 0 0 hsla(var(--profit), 0.7)" },
          "70%": { boxShadow: "0 0 0 10px hsla(var(--profit), 0)" },
          "100%": { boxShadow: "0 0 0 0 hsla(var(--profit), 0)" },
        },
        "pulse-sell": {
          "0%": { boxShadow: "0 0 0 0 hsla(var(--loss), 0.7)" },
          "70%": { boxShadow: "0 0 0 10px hsla(var(--loss), 0)" },
          "100%": { boxShadow: "0 0 0 0 hsla(var(--loss), 0)" },
        },
        "count-up": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(-100%)" },
        },
        "count-down": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(100%)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.2s ease-out",
        "fade-out": "fade-out 0.2s ease-out",
        "slide-in-from-right": "slide-in-from-right 0.3s ease-out",
        "slide-out-to-right": "slide-out-to-right 0.3s ease-out",
        "price-flash-up": "price-flash-up 0.6s ease-out",
        "price-flash-down": "price-flash-down 0.6s ease-out",
        "pulse-buy": "pulse-buy 1.5s infinite",
        "pulse-sell": "pulse-sell 1.5s infinite",
        "count-up": "count-up 0.3s ease-out",
        "count-down": "count-down 0.3s ease-out",
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
} 