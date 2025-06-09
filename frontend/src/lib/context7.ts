/**
 * Context7 API Client
 * 
 * This is a lightweight client for interacting with Context7 documentation services.
 * It provides a way to fetch library documentation and resolve library IDs.
 */

interface Context7Response {
  content: string;
  libraryId: string;
  tokens?: number;
}

interface LibrarySearchResult {
  libraryId: string;
  name: string;
  description: string;
  score: number;
  documentationCoverage: number;
  trustScore: number;
}

/**
 * Context7 client for accessing documentation
 */
export class Context7Client {
  private baseUrl: string;

  constructor(baseUrl = '/api/context7') {
    this.baseUrl = baseUrl;
  }

  /**
   * Resolves a library name to a Context7-compatible library ID
   */
  async resolveLibraryId(libraryName: string): Promise<LibrarySearchResult[]> {
    try {
      // In a real implementation, this would make an API call
      // For now, return a mock result to prevent runtime errors
      return [
        {
          libraryId: `/org/${libraryName.toLowerCase().replace(/\s+/g, '-')}`,
          name: libraryName,
          description: `Documentation for ${libraryName}`,
          score: 0.95,
          documentationCoverage: 85,
          trustScore: 9
        }
      ];
    } catch (error) {
      console.error('Error resolving library ID:', error);
      return [];
    }
  }

  /**
   * Fetches documentation for a specific library
   */
  async getLibraryDocs(
    libraryId: string,
    options?: { tokens?: number; topic?: string }
  ): Promise<Context7Response> {
    try {
      // In a real implementation, this would make an API call
      // For now, return a mock result to prevent runtime errors
      return {
        content: `# ${libraryId.split('/').pop() || 'Documentation'}\n\nPlaceholder documentation content for ${libraryId}`,
        libraryId,
        tokens: options?.tokens || 1000
      };
    } catch (error) {
      console.error('Error fetching library docs:', error);
      return {
        content: 'Error fetching documentation',
        libraryId,
        tokens: 0
      };
    }
  }
}

// Export a singleton instance for use throughout the app
export const context7 = new Context7Client();

// Default export for direct import
export default context7; 