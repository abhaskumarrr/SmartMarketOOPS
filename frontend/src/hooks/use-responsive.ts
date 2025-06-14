import { useEffect, useState, useCallback } from 'react'

// Align with Tailwind's default breakpoints
export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
}

type Breakpoint = keyof typeof breakpoints
type BreakpointState = Record<Breakpoint, boolean>

interface ResponsiveUtilities {
  breakpoints: BreakpointState;
  atLeast: (breakpoint: Breakpoint) => boolean;
  smallerThan: (breakpoint: Breakpoint) => boolean;
  at: (breakpoint: Breakpoint) => boolean;
  between: (min: Breakpoint, max: Breakpoint) => boolean;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  width: number | null;
}

/**
 * Hook that returns the current active breakpoints and responsive utilities
 * Enhanced for Tailwind CSS v4 compatibility
 */
export function useBreakpoints(): ResponsiveUtilities {
  const [width, setWidth] = useState<number | null>(null)
  const [state, setState] = useState<BreakpointState>({
    sm: false,
    md: false,
    lg: false,
    xl: false,
    '2xl': false,
  })

  // Update all breakpoints based on current window size
  const updateBreakpoints = useCallback(() => {
    if (typeof window === 'undefined') return

    const newWidth = window.innerWidth
    setWidth(newWidth)

    const newState = Object.entries(breakpoints).reduce(
      (acc, [key, breakpoint]) => ({
        ...acc,
        [key]: newWidth >= breakpoint,
      }),
      {} as BreakpointState
    )

    setState(newState)
  }, [])

  // Setup listeners for window resize
  useEffect(() => {
    if (typeof window === 'undefined') return

    // Initial calculation
    updateBreakpoints()

    // Handle resize events
    const handleResize = () => {
      requestAnimationFrame(updateBreakpoints)
    }

    window.addEventListener('resize', handleResize)
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [updateBreakpoints])

  // Check if viewport is at least the specified breakpoint
  const atLeast = useCallback(
    (breakpoint: Breakpoint) => state[breakpoint],
    [state]
  )

  // Check if viewport is smaller than the specified breakpoint
  const smallerThan = useCallback(
    (breakpoint: Breakpoint) => {
      const breakpointValue = breakpoints[breakpoint]
      return width !== null && width < breakpointValue
    },
    [width]
  )

  // Check if the viewport is exactly at the specified breakpoint range
  const at = useCallback(
    (breakpoint: Breakpoint) => {
      const keys = Object.keys(breakpoints) as Breakpoint[]
      const index = keys.indexOf(breakpoint)
      const isAtLeastCurrent = state[breakpoint]
      
      if (index === keys.length - 1) return isAtLeastCurrent
      
      const nextBreakpoint = keys[index + 1]
      const isLessThanNext = !state[nextBreakpoint]
      
      return isAtLeastCurrent && isLessThanNext
    },
    [state]
  )

  // Check if the viewport is between two breakpoints (inclusive of min, exclusive of max)
  const between = useCallback(
    (min: Breakpoint, max: Breakpoint) => {
      return state[min] && !state[max]
    },
    [state]
  )

  // Derived values for common device categories
  const isMobile = !state.md // < 768px
  const isTablet = state.md && !state.lg // >= 768px and < 1024px
  const isDesktop = state.lg // >= 1024px

  return {
    breakpoints: state,
    atLeast,
    smallerThan,
    at,
    between,
    isMobile,
    isTablet,
    isDesktop,
    width,
  }
}

/**
 * Simpler hook that just returns if the current viewport is considered "mobile"
 * Mobile is defined as smaller than the "md" breakpoint (768px)
 */
export function useIsMobile() {
  const { isMobile } = useBreakpoints()
  return isMobile
} 