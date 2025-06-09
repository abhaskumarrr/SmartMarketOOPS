import { NextRequest, NextResponse } from 'next/server';

// In-memory storage for thinking chains (would use a database in production)
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
  id: string;
  steps: ThoughtStep[];
  currentStep: number;
  completed: boolean;
  createdAt: Date;
  updatedAt: Date;
}

// Mock storage
const thinkingChains = new Map<string, ThinkingChain>();

/**
 * Create a new thinking chain
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { thought, totalThoughts = 5 } = body;
    
    if (!thought) {
      return NextResponse.json(
        { error: 'Initial thought is required' },
        { status: 400 }
      );
    }
    
    const chainId = `chain_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    
    const newChain: ThinkingChain = {
      id: chainId,
      steps: [
        {
          thought,
          thoughtNumber: 1,
          totalThoughts,
          nextThoughtNeeded: true
        }
      ],
      currentStep: 1,
      completed: false,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    thinkingChains.set(chainId, newChain);
    
    return NextResponse.json({
      chainId,
      step: newChain.steps[0],
      message: 'Thinking chain created successfully'
    });
  } catch (error) {
    console.error('Error creating thinking chain:', error);
    return NextResponse.json(
      { error: 'Failed to create thinking chain' },
      { status: 500 }
    );
  }
}

/**
 * Add a thought to an existing chain or get chain details
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      chainId, 
      thought, 
      isRevision, 
      revisesThought,
      branchFromThought,
      branchId,
      needsMoreThoughts,
      nextThoughtNeeded
    } = body;
    
    if (!chainId) {
      return NextResponse.json(
        { error: 'Chain ID is required' },
        { status: 400 }
      );
    }
    
    const chain = thinkingChains.get(chainId);
    if (!chain) {
      return NextResponse.json(
        { error: 'Thinking chain not found' },
        { status: 404 }
      );
    }
    
    if (!thought) {
      // If no thought provided, just return the chain
      return NextResponse.json({ chain });
    }
    
    const nextThoughtNumber = chain.currentStep + 1;
    const currentTotalThoughts = chain.steps[0]?.totalThoughts || 5;
    
    // Determine if we need more thoughts
    const willNeedMoreThoughts = nextThoughtNeeded ?? 
      (nextThoughtNumber < currentTotalThoughts || needsMoreThoughts === true);
    
    const newThought: ThoughtStep = {
      thought,
      thoughtNumber: nextThoughtNumber,
      totalThoughts: currentTotalThoughts,
      nextThoughtNeeded: willNeedMoreThoughts
    };
    
    if (isRevision) newThought.isRevision = true;
    if (revisesThought) newThought.revisesThought = revisesThought;
    if (branchFromThought) newThought.branchFromThought = branchFromThought;
    if (branchId) newThought.branchId = branchId;
    if (needsMoreThoughts !== undefined) newThought.needsMoreThoughts = needsMoreThoughts;
    
    chain.steps.push(newThought);
    chain.currentStep = nextThoughtNumber;
    chain.completed = !willNeedMoreThoughts;
    chain.updatedAt = new Date();
    
    return NextResponse.json({
      chainId,
      step: newThought,
      chain,
      message: 'Thought added successfully'
    });
  } catch (error) {
    console.error('Error adding thought:', error);
    return NextResponse.json(
      { error: 'Failed to add thought' },
      { status: 500 }
    );
  }
}

/**
 * Get a thinking chain by ID
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const chainId = searchParams.get('chainId');
    
    if (!chainId) {
      // Return all chains if no ID provided
      return NextResponse.json({
        chains: Array.from(thinkingChains.values())
      });
    }
    
    const chain = thinkingChains.get(chainId);
    if (!chain) {
      return NextResponse.json(
        { error: 'Thinking chain not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({ chain });
  } catch (error) {
    console.error('Error fetching thinking chain:', error);
    return NextResponse.json(
      { error: 'Failed to fetch thinking chain' },
      { status: 500 }
    );
  }
} 