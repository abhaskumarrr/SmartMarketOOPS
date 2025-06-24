# MCP Servers Setup Guide

## Overview

This guide will help you set up all the necessary API keys and environment variables for the MCP servers installed in SmartMarketOOPS.

## Required API Keys

### 1. Task Master AI (Multiple AI Providers)

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add to your `.env` file:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
```

#### Anthropic API Key (Claude)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Add to your `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

#### OpenRouter API Key (Multi-model access)
1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to Keys section
4. Create a new API key
5. Add to your `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
```

#### Google AI API Key (Gemini)
1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Sign up or log in
3. Create a new API key
4. Add to your `.env` file:
```bash
GOOGLE_API_KEY=your-google-ai-key-here
```

#### Perplexity API Key
1. Go to [Perplexity AI](https://www.perplexity.ai/)
2. Sign up for API access
3. Create a new API key
4. Add to your `.env` file:
```bash
PERPLEXITY_API_KEY=pplx-your-perplexity-key-here
```

### 2. Brave Search API Key
1. Go to [Brave Search API](https://api.search.brave.com/)
2. Sign up for an account
3. Create a new API key
4. Add to your `.env` file:
```bash
BRAVE_API_KEY=your-brave-search-key-here
```

### 3. GitHub Personal Access Token
1. Go to [GitHub Settings](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select appropriate scopes:
   - `repo` (for repository access)
   - `read:org` (for organization access)
   - `workflow` (for GitHub Actions)
4. Generate and copy the token
5. Add to your `.env` file:
```bash
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your-github-token-here
```

## Optional API Keys

### XAI API Key (Grok)
1. Go to [xAI Console](https://console.x.ai/)
2. Sign up or log in
3. Create a new API key
4. Add to your `.env` file:
```bash
XAI_API_KEY=xai-your-xai-key-here
```

### Mistral API Key
1. Go to [Mistral AI](https://console.mistral.ai/)
2. Sign up or log in
3. Create a new API key
4. Add to your `.env` file:
```bash
MISTRAL_API_KEY=your-mistral-key-here
```

### Azure OpenAI API Key
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create an Azure OpenAI resource
3. Get the API key from the resource
4. Add to your `.env` file:
```bash
AZURE_OPENAI_API_KEY=your-azure-openai-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### Ollama API Key (Local AI)
1. Install [Ollama](https://ollama.ai/)
2. Run Ollama locally
3. Add to your `.env` file:
```bash
OLLAMA_API_KEY=your-ollama-key-here
OLLAMA_BASE_URL=http://localhost:11434
```

## Database Configuration

### PostgreSQL Connection
Ensure your PostgreSQL database is running and accessible:
```bash
POSTGRES_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5432/smartmarket
```

## Complete .env File Template

Create or update your `.env` file with the following template:

```bash
# ============================================================================
# MCP SERVERS CONFIGURATION
# ============================================================================

# Task Master AI - Multiple AI Providers
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
GOOGLE_API_KEY=your-google-ai-key-here
PERPLEXITY_API_KEY=pplx-your-perplexity-key-here
XAI_API_KEY=xai-your-xai-key-here
MISTRAL_API_KEY=your-mistral-key-here
AZURE_OPENAI_API_KEY=your-azure-openai-key-here
OLLAMA_API_KEY=your-ollama-key-here

# Brave Search API
BRAVE_API_KEY=your-brave-search-key-here

# GitHub API
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your-github-token-here

# Database Configuration
POSTGRES_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5432/smartmarket

# ============================================================================
# EXISTING SMARTMARKETOOPS CONFIGURATION
# ============================================================================

# System Configuration
NODE_ENV=development
TRADING_MODE=test
FORCE_TESTNET=true
PORT=3006
FRONTEND_PORT=3000
ML_PORT=3002

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/smartmarket
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=smartmarket
POSTGRES_PORT=5432

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# QuestDB Configuration
QUESTDB_HOST=localhost
QUESTDB_PORT=9000
QUESTDB_HTTP_PORT=9009

# Security & Authentication
JWT_SECRET=your-jwt-secret-key-here-change-this-in-production
JWT_EXPIRES_IN=1h
JWT_REFRESH_SECRET=your-refresh-jwt-secret-key-here-change-this
COOKIE_SECRET=your-cookie-secret-key-here-change-this
COOKIE_DOMAIN=localhost

# Encryption
ENCRYPTION_MASTER_KEY=your-32-char-encryption-key-here-change-this-in-production
ENCRYPTION_KEY_SECONDARY=your-secondary-encryption-key-here

# CORS & Client Configuration
CLIENT_URL=http://localhost:3000
CORS_ORIGIN=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:3006

# Delta Exchange API Configuration
DELTA_EXCHANGE_API_KEY=your-delta-exchange-api-key
DELTA_EXCHANGE_API_SECRET=your-delta-exchange-api-secret
DELTA_EXCHANGE_TESTNET=true
DELTA_EXCHANGE_BASE_URL=https://cdn-ind.testnet.deltaex.org

# ML Configuration
ML_API_URL=http://localhost:3002/api
ML_API_KEY=your-ml-api-key-here

# Monitoring & Logging
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true
```

## Testing Your Setup

After configuring your environment variables, test your MCP servers:

```bash
# Run the MCP servers test script
node scripts/test-mcp-servers.js
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use different API keys for development and production**
3. **Regularly rotate your API keys**
4. **Set up API key usage alerts where available**
5. **Use environment-specific .env files**

## Troubleshooting

### Common Issues

1. **API Key Format Errors**
   - Ensure API keys are copied exactly as provided
   - Check for extra spaces or characters

2. **Rate Limiting**
   - Some APIs have rate limits for free tiers
   - Monitor your usage and upgrade plans as needed

3. **Network Issues**
   - Ensure your firewall allows outbound connections
   - Check if your organization blocks certain APIs

4. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check connection string format
   - Ensure database exists

### Getting Help

1. Check the specific API provider's documentation
2. Review the MCP server logs in Cursor IDE
3. Use the test script to identify specific issues
4. Check the project's GitHub issues for known problems

## Cost Considerations

### Free Tiers Available
- OpenAI: Limited free credits
- Anthropic: Limited free usage
- Google AI: Free tier available
- Brave Search: Free tier with limits
- GitHub: Free for public repositories

### Paid Services
- Most AI providers offer pay-per-use pricing
- Consider setting up billing alerts
- Monitor usage to avoid unexpected charges

## Next Steps

Once your MCP servers are configured:

1. **Test the integration** with the test script
2. **Explore the capabilities** of each server
3. **Integrate with your development workflow**
4. **Set up monitoring** for API usage and costs
5. **Document your specific use cases** for each server

## Conclusion

With these MCP servers properly configured, you'll have access to powerful AI assistance, database operations, web search, and automation capabilities that will significantly enhance your SmartMarketOOPS development experience.