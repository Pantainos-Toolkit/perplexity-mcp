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


def _chat_completion(
    query: str,
    model: Literal["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"],
    search_mode: Optional[Literal["web", "academic", "sec"]] = None,
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 5000,
    search_context_size: Literal["low", "medium", "high"] = "medium",
    system_prompt: Optional[str] = "Be concise and factual. Cite sources. Avoid speculation.",
) -> str:
    """Helper function for chat completion API calls."""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "web_search_options": {
                "search_context_size": search_context_size
            }
        }

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
def search(
    query: str,
    max_results: int = 10,
    max_tokens_per_page: int = 1024,
    country: Optional[str] = None,
) -> str:
    """
    **PREFER THIS FIRST** - Find and evaluate sources yourself. Returns URLs, titles, and snippets.

    Args:
        query: Search query
        max_results: Max results (1-20, default: 10)
        max_tokens_per_page: Max tokens per page (default: 1024)
        country: Two-letter country code to filter results (e.g., 'US', 'GB', 'DE')

    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "max_results": max_results,
            "max_tokens_per_page": max_tokens_per_page,
        }

        if country:
            payload["country"] = country

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
    sources: Literal["web", "sec", "academic"] = "web",
    scope: Literal["standard", "extensive"] = "standard",
    thoroughness: Literal["quick", "detailed"] = "quick",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_related_questions: bool = False,
    max_tokens: int = 5000,
) -> str:
    """
    Get AI-synthesized answers with web-grounded search.

    Args:
        query: Your question
        sources: Source type - 'web' (general), 'sec' (financial filings), 'academic' (scholarly)
        scope: Search breadth - 'standard' (normal) or 'extensive' (2x more sources)
        thoroughness: Content extraction - 'quick' (recommended) or 'detailed' (only if absolutely needed, prefer scope='extensive' instead)
        recency: Recent content - 'day', 'week', or 'month'
        domain_filter: Filter by domain. Use '-' to exclude. Examples: ['github.com'], ['-reddit.com']
        return_related_questions: Get follow-up questions
        max_tokens: Max response length (default: 5000)

    Returns:
        AI-synthesized answer with citations
    """
    # Map scope to model
    model = "sonar" if scope == "standard" else "sonar-pro"

    # Map thoroughness to search_context_size
    search_context_size = "low" if thoroughness == "quick" else "high"

    return _chat_completion(
        query=query,
        model=model,
        search_mode=sources,
        recency=recency,
        domain_filter=domain_filter,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
        search_context_size=search_context_size,
        system_prompt="Be concise and factual. Cite sources. Avoid speculation.",
    )


@mcp.tool
def ask_reasoning(
    query: str,
    scope: Literal["standard", "extensive"] = "standard",
    thoroughness: Literal["quick", "detailed"] = "quick",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_related_questions: bool = False,
    max_tokens: int = 5000,
) -> str:
    """
    Get answers with explicit step-by-step reasoning.

    Shows reasoning process with <think> sections. Use for multi-step problems and complex reasoning.

    Args:
        query: Your question
        scope: Search breadth - 'standard' (normal) or 'extensive' (2x more sources, deeper reasoning)
        thoroughness: Content extraction - 'quick' (recommended) or 'detailed' (only if absolutely needed, prefer scope='extensive' instead)
        recency: Recent content - 'day', 'week', or 'month'
        domain_filter: Filter by domain. Use '-' to exclude. Examples: ['github.com'], ['-reddit.com']
        return_related_questions: Get follow-up questions
        max_tokens: Max response length (default: 5000)

    Returns:
        Answer with explicit reasoning steps and citations
    """
    # Map scope to model
    model = "sonar-reasoning" if scope == "standard" else "sonar-reasoning-pro"

    # Map thoroughness to search_context_size
    search_context_size = "low" if thoroughness == "quick" else "high"

    return _chat_completion(
        query=query,
        model=model,
        search_mode="web",
        recency=recency,
        domain_filter=domain_filter,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
        search_context_size=search_context_size,
        system_prompt="Be concise and factual. Cite sources. Avoid speculation.",
    )


if __name__ == "__main__":
    # Run the server - works for both local (stdio) and cloud (HTTP)
    mcp.run()
