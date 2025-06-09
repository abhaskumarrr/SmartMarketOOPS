/**
 * Sequential Thinking Implementation
 * 
 * This module provides a client for working with the Sequential Thinking API,
 * which helps break down complex problems through structured thinking steps.
 */

interface ThoughtStep {
  thought: string;
  thoughtNumber: number;
  totalThoughts: number;
  nextThoughtNeeded: boolean;
  isRevision?: boolean;
  revisesThought?: number;
  branchFromThought?: number;
  branchId?: string;
  needsMoreThoughts?: boolean;
}

interface ThinkingChain {
  steps: ThoughtStep[];
  currentStep: number;
  completed: boolean;
}

/**
 * Sequential Thinking client for step-by-step problem solving
 */
export class SequentialThinkingClient {
  private baseUrl: string;
  private chains: Map<string, ThinkingChain>;

  constructor(baseUrl = '/api/sequential-thinking') {
    this.baseUrl = baseUrl;
    this.chains = new Map();
  }

  /**
   * Creates a new thinking chain for a problem
   */
  createThinkingChain(initialThought: string, totalThoughts = 5): string {
    const chainId = `chain_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    
    this.chains.set(chainId, {
      steps: [
        {
          thought: initialThought,
          thoughtNumber: 1,
          totalThoughts,
          nextThoughtNeeded: true
        }
      ],
      currentStep: 1,
      completed: false
    });
    
    return chainId;
  }

  /**
   * Adds a new thought to an existing chain
   */
  addThought(
    chainId: string, 
    thoughtText: string, 
    options?: {
      isRevision?: boolean;
      revisesThought?: number;
      branchFromThought?: number;
      branchId?: string;
      needsMoreThoughts?: boolean;
      nextThoughtNeeded?: boolean;
    }
  ): ThoughtStep | null {
    const chain = this.chains.get(chainId);
    if (!chain) return null;
    
    const nextThoughtNumber = chain.currentStep + 1;
    const currentTotalThoughts = chain.steps[0]?.totalThoughts || 5;
    
    // If this is the last thought and we don't need more, mark as complete
    const nextThoughtNeeded = options?.nextThoughtNeeded ?? 
      (nextThoughtNumber < currentTotalThoughts || options?.needsMoreThoughts === true);
    
    const newThought: ThoughtStep = {
      thought: thoughtText,
      thoughtNumber: nextThoughtNumber,
      totalThoughts: currentTotalThoughts,
      nextThoughtNeeded,
      ...options
    };
    
    chain.steps.push(newThought);
    chain.currentStep = nextThoughtNumber;
    chain.completed = !nextThoughtNeeded;
    
    return newThought;
  }

  /**
   * Gets the full thinking chain by ID
   */
  getThinkingChain(chainId: string): ThinkingChain | null {
    return this.chains.get(chainId) || null;
  }

  /**
   * Gets all thinking chains
   */
  getAllThinkingChains(): Map<string, ThinkingChain> {
    return this.chains;
  }
}

// Export a singleton instance for use throughout the app
export const sequentialThinking = new SequentialThinkingClient();

// Default export for direct import
export default sequentialThinking; 