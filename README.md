# Recreating-the-Bing-Chatbot

## Introduction
While the Large Language Models (LLMs) possess impressive capabilities, they have certain limitations that can present challenges when deploying them in a production environment. The hallucination problem makes them answer certain questions wrongly with high confidence. This issue can be attributed to various factors, one of which is that their training process has a cut-off date. So, these models do not have access to events preceding that date.

A workaround approach is to present the required information to the model and leverage its reasoning capability to find/extract the answer. Furthermore, it is possible to present the top-matched results a search engine returns as the context for a user’s query.

This lesson will explore the idea of finding the best articles from the Internet as the context for a chatbot to find the correct answer. We will use LangChain’s integration with Google Search API and the Newspaper library to extract the stories from search results. This is followed by choosing and using the most relevant options in the prompt.

Notice that the same pipeline could be done with the Bing API, but we’ll use the Google Search API in this project because it is used in other lessons of this course, thus avoiding creating several keys for the same functionality. Please refer to the following [tutorial](https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e) (or [Bing Web Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api) for direct access) on obtaining the Bing Subscription Key and using the LangChain Bing search [wrapper](https://python.langchain.com/docs/integrations/tools/bing_search?highlight=Bing)https://python.langchain.com/docs/integrations/tools/bing_search?highlight=Bing.

What we are going to do is explained in the following diagram.

<img align="center" src="bcb.avif" alt="banner">
