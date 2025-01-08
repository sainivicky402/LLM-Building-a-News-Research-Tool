import os
from langchain_openai import OpenAI  # Updated import
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate  # Corrected import
from newsapi import NewsApiClient
from typing import List, Dict
from pydantic import BaseModel  # Updated import

# Load API keys from environment variables for security
openai_api_key = os.getenv('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')  # Use the environment variable name
newsapi_key = os.getenv('d70b0432bebb46e98a2f8b5555f5a994')  

# Check if API keys are loaded
if not openai_api_key or not newsapi_key:
    raise ValueError("API keys for OpenAI and NewsAPI must be set in environment variables.")

# Initialize OpenAI API
openai = OpenAI(api_key=openai_api_key)

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=newsapi_key)

def create_prompt_template() -> PromptTemplate:
    """Creates a prompt template for summarization."""
    template = """
    You are an AI assistant helping an equity research analyst. Given
    the following query and the provided news article summaries, provide
    an overall summary.
    Query: {query}
    Summaries: {summaries}
    """
    return PromptTemplate(template=template, input_variables=['query', 'summaries'])

def get_news_articles(query: str) -> List[Dict]:
    """Fetches relevant news articles using NewsAPI."""
    try:
        response = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
        return response.get('articles', [])
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

def summarize_articles(articles: List[Dict]) -> str:
    """Creates a combined summary from article descriptions."""
    summaries = [article['description'] for article in articles if 'description' in article]
    return ' '.join(summaries)

def get_summary(query: str) -> str:
    """Retrieves news articles and generates a combined summary."""
    articles = get_news_articles(query)
    if not articles:
        return "No articles found for the given query."
    
    summaries = summarize_articles(articles)
    prompt = create_prompt_template()
    llm_chain = LLMChain(prompt=prompt, llm=openai)
    
    return llm_chain.run({'query': query, 'summaries': summaries})

# Example usage
if __name__ == "__main__":
    query = "latest technology trends"
    summary = get_summary(query)
    print(summary)  # Removed extra parenthesis
