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
    model: Literal["sonar", "sonar-pro"],
    search_mode: Optional[Literal["web", "academic", "sec"]] = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 500,
) -> str:
    """Helper function for chat completion API calls."""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
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
    domain_filter: Optional[list[str]] = None,
) -> str:
    """
    **PREFER THIS FIRST** - Find and evaluate sources yourself. Returns URLs, titles, and snippets so you can assess quality and make your own conclusions.

    Start here to ground yourself before using ask tools. The librarian shows you the shelf - you decide what matters.

    Use when:
    - Starting research on any topic (preferred first step)
    - You need to assess source quality or credibility
    - Looking for specific documents, reports, or official pages
    - You want control over what information to trust
    - Research requires seeing multiple perspectives

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
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 500,
) -> str:
    """
    Get a direct answer from general web search. Fast and cost-effective.

    **Consider using 'search' first** to ground yourself with sources before getting synthesized answers.

    Perplexity reads sources and gives you the conclusion - the librarian reads the books and tells you the answer.

    Use for:
    - Quick facts and definitions ("What is X?")
    - General explanations and comparisons ("How does X compare to Y?")
    - Recent news and developments (combine with recency filter)
    - Most general knowledge questions

    NOT for financial filings or academic papers - use ask_sec or ask_academic instead.

    Args:
        query: Your question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (thorough)
        recency: Focus on recent results - 'day', 'week', or 'month'
        domain_filter: Include/exclude domains (e.g., ['wikipedia.org'] or ['-reddit.com'])
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 500)

    Returns:
        AI-synthesized answer with citations
    """
    return _chat_completion(
        query=query,
        model="sonar",
        search_mode="web",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


@mcp.tool
def ask_more(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 1000,
) -> str:
    """
    Like 'ask' but significantly MORE comprehensive and detailed for general web questions. Slower and more expensive.

    Use when: Standard 'ask' doesn't provide enough depth for your general web research question.

    Args:
        query: Your complex question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (most thorough)
        recency: Focus on recent results - 'day', 'week', or 'month'
        domain_filter: Include/exclude domains
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 1000)

    Returns:
        Comprehensive AI-synthesized answer with detailed citations
    """
    return _chat_completion(
        query=query,
        model="sonar-pro",
        search_mode="web",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


@mcp.tool
def ask_sec(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 500,
) -> str:
    """
    Get answers from SEC filings and financial regulatory documents.

    **Tip**: Consider using 'search' first to find the relevant filings, then use this for synthesis.

    Use for company financials:
    - Earnings reports and quarterly results
    - Revenue, margins, cash flow, profitability
    - 10-K annual reports, 10-Q quarterly filings, 8-K current reports
    - Financial performance analysis
    - Regulatory disclosures

    Examples:
    - "Tesla Q4 2024 earnings and revenue"
    - "Apple cash flow analysis 2024"
    - "Microsoft quarterly results latest"
    - "Pure Storage financial performance 2025"

    Args:
        query: Your financial question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (thorough)
        recency: Focus on recent filings - 'day', 'week', or 'month'
        domain_filter: Include/exclude domains
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 500)

    Returns:
        AI-synthesized answer from SEC filings with citations
    """
    return _chat_completion(
        query=query,
        model="sonar",
        search_mode="sec",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


@mcp.tool
def ask_sec_more(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 1000,
) -> str:
    """
    Like 'ask_sec' but MORE comprehensive financial analysis. Slower and more expensive.

    Use when: Standard 'ask_sec' doesn't provide enough depth for complex financial analysis.

    Args:
        query: Your complex financial question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (most thorough)
        recency: Focus on recent filings
        domain_filter: Include/exclude domains
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 1000)

    Returns:
        Comprehensive financial analysis from SEC filings
    """
    return _chat_completion(
        query=query,
        model="sonar-pro",
        search_mode="sec",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


@mcp.tool
def ask_academic(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 500,
) -> str:
    """
    Get answers from scholarly papers and academic research publications.

    **Tip**: Consider using 'search' first to find relevant papers, then use this for synthesis.

    Use for research questions:
    - Scientific studies and research findings
    - Academic papers and peer-reviewed publications
    - Scholarly analysis and literature reviews
    - Research methodologies and results
    - Citations to academic sources

    Examples:
    - "Latest research on transformer neural networks"
    - "Climate change impact studies 2024"
    - "Quantum computing breakthroughs"

    Args:
        query: Your research question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (thorough)
        recency: Focus on recent research - 'day', 'week', or 'month'
        domain_filter: Include/exclude domains
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 500)

    Returns:
        AI-synthesized answer from academic sources with citations
    """
    return _chat_completion(
        query=query,
        model="sonar",
        search_mode="academic",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


@mcp.tool
def ask_academic_more(
    query: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    recency: Optional[Literal["day", "week", "month"]] = None,
    domain_filter: Optional[list[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    max_tokens: int = 1000,
) -> str:
    """
    Like 'ask_academic' but MORE comprehensive academic research. Slower and more expensive.

    Use when: Standard 'ask_academic' doesn't provide enough depth for complex research questions.

    Args:
        query: Your complex research question
        reasoning_effort: 'low' (fastest), 'medium' (balanced, default), 'high' (most thorough)
        recency: Focus on recent research
        domain_filter: Include/exclude domains
        return_images: Include related images
        return_related_questions: Get follow-up question suggestions
        max_tokens: Maximum tokens in response (default: 1000)

    Returns:
        Comprehensive academic research synthesis
    """
    return _chat_completion(
        query=query,
        model="sonar-pro",
        search_mode="academic",
        reasoning_effort=reasoning_effort,
        recency=recency,
        domain_filter=domain_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    # Run the server - works for both local (stdio) and cloud (HTTP)
    mcp.run()