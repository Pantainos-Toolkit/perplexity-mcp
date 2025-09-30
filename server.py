"""
Perplexity MCP Server

A FastMCP server that provides web search and grounded AI answers using Perplexity's API.

Workflow:
1. search - Ground yourself first by finding sources
2. ask - Get AI-synthesized answers from those sources
3. ask_more - Dig deeper with more comprehensive analysis
"""

import os
from typing import Literal, Optional
from fastmcp import FastMCP
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Perplexity Research")

# Get API key from environment
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise ValueError(
        "PERPLEXITY_API_KEY environment variable is required. "
        "Get your API key from https://www.perplexity.ai/settings/api"
    )

# API configuration
PERPLEXITY_API_BASE = "https://api.perplexity.ai"
SEARCH_ENDPOINT = f"{PERPLEXITY_API_BASE}/search"
CHAT_ENDPOINT = f"{PERPLEXITY_API_BASE}/chat/completions"


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string."""
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. {result.get('title', 'No title')}")
        formatted.append(f"   URL: {result.get('url', 'No URL')}")
        if snippet := result.get('snippet'):
            formatted.append(f"   {snippet}")
        formatted.append("")
    return "\n".join(formatted)


def format_chat_response(response: dict) -> str:
    """Format chat completion response with citations."""
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    output = [content]

    # Add citations if available
    if citations := response.get("citations"):
        output.append("\n\n📚 Sources:")
        for i, citation in enumerate(citations, 1):
            output.append(f"{i}. {citation}")

    # Add images if available
    if images := response.get("images"):
        output.append("\n\n🖼️ Related Images:")
        for img_url in images[:5]:  # Limit to 5 images
            output.append(f"- {img_url}")

    # Add related questions if available
    if related := response.get("related_questions"):
        output.append("\n\n❓ Related Questions:")
        for question in related:
            output.append(f"- {question}")

    return "\n".join(output)


@mcp.tool
def search(
    query: str,
    max_results: int = 10,
    domain_filter: Optional[list[str]] = None,
) -> str:
    """
    Find and evaluate sources yourself - returns URLs, titles, and snippets so you can assess quality and make your own conclusions.

    Use when:
    - You need to assess source quality or credibility
    - Looking for specific documents, reports, or official pages
    - You want control over what information to trust
    - Research requires seeing multiple perspectives

    Think: The librarian shows you the shelf - you decide what matters.

    Examples:
    - Find official documentation sites
    - Evaluate multiple sources on a controversial topic
    - Locate specific reports or papers to cite
    - Filter to trusted domains like .edu or .gov

    Args:
        query: Search query to find relevant sources
        max_results: Maximum number of results to return (default: 10)
        domain_filter: List of domains to include (e.g., ['wikipedia.org']) or
                      exclude (prefix with '-', e.g., ['-reddit.com'])

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "max_results": max_results,
        }

        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        with httpx.Client(timeout=30.0) as client:
            response = client.post(SEARCH_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not results:
            return "No search results found."

        return format_search_results(results)

    except httpx.HTTPStatusError as e:
        return f"Search API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Search failed: {str(e)}"


@mcp.tool
def ask(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    search_mode: Optional[Literal["web", "academic", "sec"]] = None,
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
) -> str:
    """
    Get a direct answer - returns AI-synthesized response with citations. Perplexity reads sources and gives you the conclusion.

    Use when:
    - You need a quick fact or explanation
    - Answer is the goal, not evaluating sources
    - You trust Perplexity's synthesis
    - Time-sensitive questions

    Think: The librarian reads the books and tells you the answer.

    Special capabilities:
    - Academic mode: Search scholarly papers and research publications
    - SEC mode: Search financial filings (10-K, 10-Q, 8-K) and regulatory documents
      → Use for: earnings, quarterly results, revenue, margins, cash flow, financial performance
    - Recency filtering: Focus on recent results (day/week/month)
    - Domain filtering: Include or exclude specific websites
    - Images: Get related visual content
    - Related questions: Receive follow-up suggestions

    Example use cases:
    - "What is X?" - quick facts and definitions
    - "How does X compare to Y?" - comparative analysis
    - Recent developments (use recency='week')
    - Academic questions (use search_mode='academic')
    - Company earnings/financials (MUST use search_mode='sec')
      Examples: "Tesla Q4 earnings", "Apple revenue 2024", "Microsoft cash flow"

    Args:
        query: Question or topic to get an AI answer about
        reasoning_effort: How much reasoning to apply - 'low' (faster),
                         'medium' (balanced, default), or 'high' (more thorough)
        search_mode: Search type - 'web' (default), 'academic' (scholarly sources),
                    or 'sec' (SEC filings)
        recency: Filter sources by time - 'day', 'week', or 'month'
        domain_filter: Include/exclude specific domains (e.g., ['wikipedia.org'] or ['-reddit.com'])
        return_images: Include related images in the response
        return_related_questions: Include follow-up question suggestions

    Returns:
        AI-generated answer with citations and optional images/related questions
    """
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
        }

        # Map reasoning_effort to actual reasoning_effort parameter
        effort_map = {"low": "low", "medium": "medium", "high": "high"}
        if reasoning_effort in effort_map:
            payload["reasoning_effort"] = effort_map[reasoning_effort]

        if search_mode:
            payload["search_mode"] = search_mode

        if recency:
            payload["search_recency_filter"] = recency

        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        if return_images:
            payload["return_images"] = True

        if return_related_questions:
            payload["return_related_questions"] = True

        with httpx.Client(timeout=60.0) as client:
            response = client.post(CHAT_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return format_chat_response(data)

    except httpx.HTTPStatusError as e:
        return f"Chat API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"


@mcp.tool
def ask_more(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    search_mode: Optional[Literal["web", "academic", "sec"]] = None,
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
) -> str:
    """
    Like 'ask' but significantly MORE comprehensive and detailed. Slower and more expensive.

    Use when: Standard 'ask' doesn't provide enough depth or you need thorough investigation.

    Same capabilities as 'ask' including:
    - SEC mode for financial filings (earnings, quarterly results, 10-K/10-Q)
    - Academic mode for scholarly research
    - All filtering options (recency, domain, images, etc.)

    Args:
        query: Complex question requiring deeper analysis
        reasoning_effort: How much reasoning to apply - 'low' (faster),
                         'medium' (balanced, default), or 'high' (most thorough)
        search_mode: Search type - 'web' (default), 'academic', or 'sec'
        recency: Filter sources by time - 'day', 'week', or 'month'
        domain_filter: Include/exclude specific domains
        return_images: Include related images in the response
        return_related_questions: Include follow-up question suggestions

    Returns:
        Comprehensive AI-generated answer with detailed citations
    """
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": query}],
        }

        # Map reasoning_effort to actual reasoning_effort parameter
        effort_map = {"low": "low", "medium": "medium", "high": "high"}
        if reasoning_effort in effort_map:
            payload["reasoning_effort"] = effort_map[reasoning_effort]

        if search_mode:
            payload["search_mode"] = search_mode

        if recency:
            payload["search_recency_filter"] = recency

        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        if return_images:
            payload["return_images"] = True

        if return_related_questions:
            payload["return_related_questions"] = True

        with httpx.Client(timeout=60.0) as client:
            response = client.post(CHAT_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return format_chat_response(data)

    except httpx.HTTPStatusError as e:
        return f"Chat API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"


if __name__ == "__main__":
    # Run the server - works for both local (stdio) and cloud (HTTP)
    mcp.run()