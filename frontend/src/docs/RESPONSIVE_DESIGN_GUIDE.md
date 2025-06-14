# SmartMarketOOPS Responsive Design Guide

This guide outlines the responsive design approach implemented in SmartMarketOOPS during Week 3, focusing on best practices for maintaining and extending the responsive components.

## Core Responsive Framework

### Breakpoint System

We use a custom hook (`useBreakpoints`) that aligns with Tailwind CSS's breakpoint system:

```typescript
// Breakpoints match Tailwind's default configuration
export const breakpoints = {
  sm: 640,   // Small devices (phones)
  md: 768,   // Medium devices (tablets)
  lg: 1024,  // Large devices (desktops)
  xl: 1280,  // Extra large devices (large desktops)
  '2xl': 1536, // Extra extra large devices
}
```

This hook provides convenient utilities like:
- `isMobile`: true when screen width < md breakpoint
- `isTablet`: true when screen width >= md and < lg breakpoint
- `isDesktop`: true when screen width >= lg breakpoint
- `atLeast(breakpoint)`: true when screen width >= specified breakpoint
- `smallerThan(breakpoint)`: true when screen width < specified breakpoint
- `between(min, max)`: true when screen width is between min and max breakpoints

### Mobile-First Approach

All components should follow a mobile-first approach, where the base styles target mobile devices and media queries add complexity for larger screens:

```jsx
// Example mobile-first pattern
<div className="
  grid grid-cols-1           // Mobile: single column
  md:grid-cols-2             // Tablet: two columns
  lg:grid-cols-3             // Desktop: three columns
  gap-2 md:gap-4 lg:gap-6    // Increasing gaps at larger sizes
">
```

## Key Responsive Components

### 1. AppSidebar

The sidebar implements an off-canvas pattern for mobile devices:
- Mobile: Hidden by default, slides in from left when toggled
- Tablet/Desktop: Always visible, expanded or collapsed

Key implementation details:
- Overlay covers the screen on mobile when sidebar is open
- Touch-friendly close button and navigation items
- Smooth transitions for opening/closing

### 2. TradingDashboard

The trading dashboard adjusts its layout based on screen size:
- Mobile: Stacked single-column layout, simplified controls
- Tablet: Two-column layout with moderate details
- Desktop: Multi-column layout with full details and features

Implementation details:
- CSS Grid with responsive column configuration
- Automatic reflow of chart and panels based on available space
- Compact mode for trade execution and position panels on smaller screens

### 3. ConfigurableDashboard

The configurable dashboard provides drag-and-drop widget management:
- Mobile: Single column layout with scrollable interface
- Tablet/Desktop: Multi-column grid with more visible widgets

Features:
- Edit mode for repositioning widgets
- Responsive widget sizing
- Persistent layout storage

## Real-Time Data Optimization

### WebSocket Performance

The `useOptimizedWebSocket` hook implements several optimizations:
- Message buffering to reduce re-renders
- Connection pooling for shared subscriptions
- Automatic reconnection with exponential backoff
- Rate limiting for high-frequency data

### Rendering Optimizations

- Charts use windowed rendering for large datasets
- Components utilize React.memo and useMemo for expensive calculations
- Data processing happens outside the render cycle

## Best Practices for Extending

1. **Always Use the Breakpoint System**
   - Leverage the `useBreakpoints` hook for consistent responsive behavior
   - Use the same breakpoint values across components

2. **Follow Mobile-First Patterns**
   - Start with mobile layout and add complexity for larger screens
   - Use Tailwind's responsive prefixes (sm:, md:, lg:, xl:) consistently

3. **Test Across Devices**
   - Verify layouts on real devices or device emulators
   - Check common screen sizes: 375px (mobile), 768px (tablet), 1024px+ (desktop)

4. **Performance Considerations**
   - Optimize real-time data processing for mobile devices
   - Consider reduced data frequency on mobile connections
   - Use simplified visualizations on smaller screens

5. **Accessibility**
   - Ensure touch targets are at least 44×44px on mobile
   - Maintain readable font sizes (minimum 16px for body text)
   - Test keyboard navigation on all screen sizes

## Troubleshooting

If a component doesn't respond correctly to screen size changes:

1. Verify you're using the `useBreakpoints` hook correctly
2. Check for conflicting fixed dimensions (px instead of relative units)
3. Inspect the component with browser dev tools at different widths
4. Confirm media queries are working as expected
5. Test with actual devices when possible 