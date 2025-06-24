# Known Issues Log

This document tracks known issues, primarily related to code quality and linting, that have been identified but not yet prioritized for immediate fixing. This log helps maintain awareness of technical debt.

---

## Frontend ESLint Issues (as of latest `npm run lint`)

The following is a summary of recurring ESLint warnings and errors across the frontend codebase. While many of these are set to "warn" during development, they should be addressed before a production release.

### 1. Unused Variables & Imports (`@typescript-eslint/no-unused-vars`)

- **Issue**: Numerous components and hooks have variables, functions, or imports that are defined but never used.
- **Example Files**: `analytics/page.tsx`, `bot/page.tsx`, `dashboard/TradingDashboard.tsx`, `components/dashboard/PositionManagementPanel.tsx`.
- **Impact**: Low. Can lead to code bloat and confusion during maintenance.
- **Action**: Periodically review and remove unused code.

### 2. Explicit `any` Type (`@typescript-eslint/no-explicit-any`)

- **Issue**: The `any` type is used in several places, which undermines the benefits of TypeScript.
- **Example Files**: `api/health/route.ts`, `hooks/useWebSocket.ts`, `lib/api.ts`.
- **Impact**: Medium. Reduces type safety and can hide potential bugs.
- **Action**: Replace `any` with more specific types or `unknown` where appropriate.

### 3. Missing React Hook Dependencies (`react-hooks/exhaustive-deps`)

- **Issue**: Several `useEffect` and `useCallback` hooks have missing dependencies in their dependency arrays.
- **Example Files**: `components/charts/TradingViewWidget.tsx`, `hooks/useDeltaExchange.ts`.
- **Impact**: Medium. Can lead to stale closures and unexpected behavior.
- **Action**: Review each warning and add the missing dependencies or refactor the hook logic.

### 4. Unescaped Entities in JSX (`react/no-unescaped-entities`)

- **Issue**: Strings containing characters like `'` or `"` are used directly in JSX without being escaped.
- **Example Files**: `analytics/page.tsx`, `bot/page.tsx`, `components/dashboard/PortfolioDisplay.tsx`.
- **Impact**: Low. Can cause rendering issues in some edge cases.
- **Action**: Replace characters with their corresponding HTML entities (e.g., `&apos;` for `'`).

### 5. Use of `<img>` instead of Next.js `<Image>` (`@next/next/no-img-element`)

- **Issue**: Standard `<img>` tags are used instead of the Next.js `<Image>` component, which provides automatic optimization.
- **Example Files**: `settings/page.tsx`, `components/ui/avatar.tsx`.
- **Impact**: Medium. Leads to slower image loading, higher bandwidth usage, and poorer Core Web Vitals scores.
- **Action**: Replace `<img>` elements with `<Image>` and configure the necessary properties.

### 6. HTML `<a>` for Internal Navigation (`@next/next/no-html-link-for-pages`)

- **Issue**: An `<a>` tag is used for navigating to an internal page instead of the Next.js `<Link>` component.
- **Example File**: `dashboard/layout.tsx`.
- **Impact**: Medium. Prevents client-side navigation, causing a full page reload and a slower user experience.
- **Action**: Replace the `<a>` tag with `<Link>`.

---

## Other Known Issues

- **Node.js Compatibility**: There appears to be a version compatibility issue preventing the installation of `react-grid-layout`. The current workaround is a custom grid implementation.
- **Development-Only ESLint Config**: The current ESLint configuration (`.eslintrc.json`) is permissive to speed up development. This should be tightened before moving to production.
- **Placeholder Components**: Some widgets in the `ConfigurableDashboard` are currently placeholders and need to be replaced with fully functional components. 