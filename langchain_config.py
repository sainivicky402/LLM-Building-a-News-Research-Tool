from langchain.chains import LLMChain  # Use LLMChain if available
from langchain_core.prompts import PromptTemplate  # For PromptTemplate
from langchain_openai import OpenAI  # For OpenAI LLM 
from newsapi import NewsApiClient
from pydantic import BaseModel  # Import BaseModel directly from pydantic

openai_api_key = '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'
newsapi_key = 'd70b0432bebb46e98a2f8b5555f5a994'

openai = OpenAI(api_key=openai_api_key)
newsapi = NewsApiClient(api_key=newsapi_key)

def get_news_articles(query):
    """Fetches relevant news articles using NewsAPI."""
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
    return articles['articles']

def summarize_articles(articles):
    """Creates a combined summary from article descriptions."""
    summaries = []
    for article in articles:
        summaries.append(article['description'])
    return ' '.join(summaries)

def get_summary(query):
    """Retrieves news articles and generates a combined summary."""
    articles = get_news_articles(query)
    summary = summarize_articles(articles)
    return summary

# Prompt template for LLM to summarize news articles
template = """
You are an AI assistant helping an equity research analyst. Given
the following query and the provided news article summaries, provide
an overall summary.
Query: {query}
Summaries: {summaries}
"""
prompt = PromptTemplate(template=template, input_variables=['query', 'summaries'])

# Create an LLMChain
llm_chain = LLMChain(prompt=prompt, llm=openai)

# Example usage
if __name__ == "__main__":
    query = "latest technology trends"
    articles = get_news_articles(query)
    summaries = summarize_articles(articles)
    
    # Run the LLM chain
    result = llm_chain.invoke({"query": query, "summaries": summaries})
    print(result)