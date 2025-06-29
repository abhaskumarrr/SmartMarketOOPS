
import { jest, describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from '@jest/globals';

declare global {
  const jest: typeof jest;
  const describe: typeof describe;
  const it: typeof it;
  const expect: typeof expect;
  const beforeEach: typeof beforeEach;
  const afterEach: typeof afterEach;
  const beforeAll: typeof beforeAll;
  const afterAll: typeof afterAll;
}
