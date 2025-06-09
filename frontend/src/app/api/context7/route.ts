import { NextRequest, NextResponse } from 'next/server';

// Mock library data for demonstration
const MOCK_LIBRARIES = [
  {
    libraryId: '/react/react',
    name: 'React',
    description: 'A JavaScript library for building user interfaces',
    score: 0.98,
    documentationCoverage: 95,
    trustScore: 10
  },
  {
    libraryId: '/vercel/next.js',
    name: 'Next.js',
    description: 'The React Framework for the Web',
    score: 0.96,
    documentationCoverage: 90,
    trustScore: 9
  }
];

/**
 * Resolve library name to libraryId
 */
export async function POST(request: NextRequest) {
  try {
    const { libraryName } = await request.json();
    
    if (!libraryName) {
      return NextResponse.json(
        { error: 'Library name is required' },
        { status: 400 }
      );
    }
    
    // In a real implementation, this would search an actual database or call an external API
    const results = MOCK_LIBRARIES.filter(lib => 
      lib.name.toLowerCase().includes(libraryName.toLowerCase())
    );
    
    return NextResponse.json({ results });
  } catch (error) {
    console.error('Error in context7 resolve endpoint:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}

/**
 * Get library documentation
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const libraryId = searchParams.get('libraryId');
    const tokens = searchParams.get('tokens') ? parseInt(searchParams.get('tokens')!) : 10000;
    const topic = searchParams.get('topic') || undefined;
    
    if (!libraryId) {
      return NextResponse.json(
        { error: 'Library ID is required' },
        { status: 400 }
      );
    }
    
    // Mock documentation content
    const content = `# ${libraryId.split('/').pop() || 'Documentation'}\n\n` +
      `This is placeholder documentation for ${libraryId}.\n\n` +
      `## Overview\n\nThis library provides functionality for your application.\n\n` +
      `## Installation\n\n\`\`\`bash\nnpm install ${libraryId.split('/').pop()}\n\`\`\`\n\n` +
      (topic ? `## ${topic}\n\nSpecific information about ${topic}.\n\n` : '');
    
    return NextResponse.json({
      content,
      libraryId,
      tokens: Math.min(tokens, content.length)
    });
  } catch (error) {
    console.error('Error in context7 docs endpoint:', error);
    return NextResponse.json(
      { error: 'Failed to fetch documentation' },
      { status: 500 }
    );
  }
} 