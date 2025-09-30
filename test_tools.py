#!/usr/bin/env python3
"""Quick test of the Perplexity MCP tools"""

import os
import sys
sys.path.insert(0, '.')

# Load environment
from dotenv import load_dotenv
load_dotenv()

import httpx

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
SEARCH_ENDPOINT = "https://api.perplexity.ai/search"
CHAT_ENDPOINT = "https://api.perplexity.ai/chat/completions"

def test_search():
    """Test the search endpoint"""
    print("🔍 Testing search endpoint...")
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": "what is fastmcp",
        "max_results": 3
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(SEARCH_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Search works! Found {len(data.get('results', []))} results")
            for i, result in enumerate(data.get('results', [])[:2], 1):
                print(f"  {i}. {result.get('title', 'No title')}")
            return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False

def test_chat():
    """Test the chat completions endpoint"""
    print("\n💬 Testing chat completions (sonar model)...")
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": "What is FastMCP in one sentence?"}]
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(CHAT_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Chat works! Response: {content[:100]}...")
            return True
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Perplexity API with your key...\n")
    search_ok = test_search()
    chat_ok = test_chat()

    print("\n" + "="*50)
    if search_ok and chat_ok:
        print("✅ All tests passed! Your MCP server should work perfectly.")
    else:
        print("❌ Some tests failed. Check your API key and permissions.")