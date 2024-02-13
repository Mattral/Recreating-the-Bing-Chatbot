# Recreating-the-Bing-Chatbot

## Introduction
While the Large Language Models (LLMs) possess impressive capabilities, they have certain limitations that can present challenges when deploying them in a production environment. The hallucination problem makes them answer certain questions wrongly with high confidence. This issue can be attributed to various factors, one of which is that their training process has a cut-off date. So, these models do not have access to events preceding that date.

A workaround approach is to present the required information to the model and leverage its reasoning capability to find/extract the answer. Furthermore, it is possible to present the top-matched results a search engine returns as the context for a user’s query.

This repo will explore the idea of finding the best articles from the Internet as the context for a chatbot to find the correct answer. We will use LangChain’s integration with Google Search API and the Newspaper library to extract the stories from search results. This is followed by choosing and using the most relevant options in the prompt.

Notice that the same pipeline could be done with the Bing API, but we’ll use the Google Search API in this project because it is used in other repos of this , thus avoiding creating several keys for the same functionality. Please refer to the following [tutorial](https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e) (or [Bing Web Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api) for direct access) on obtaining the Bing Subscription Key and using the LangChain Bing search [wrapper](https://python.langchain.com/docs/integrations/tools/bing_search?highlight=Bing)https://python.langchain.com/docs/integrations/tools/bing_search?highlight=Bing.

What we are going to do is explained in the following diagram.

<img align="center" src="bcb.avif" alt="banner">

The user query is used to extract relevant articles using a search engine (e.g. Bing or Google Search), which are then split into chunks. We then compute the embeddings of each chunk, rank them by cosine similarity with respect to the embedding of the query, and put the most relevant chunks into a prompt to generate the final answer, while also keeping track of the sources.

## Ask Trending Questions
Let’s start this repo by seeing an example. The following piece must be familiar by now. It uses the OpenAI GPT-3.5-turbo model to create an assistant to answer questions. We will ask the model to name the latest Fast & Furious movie, released recently. Therefore, the model couldn’t have seen the answer during the training. Remember to install the required packages with the following command: pip install langchain deeplake openai tiktoken.

```
The user query is used to extract relevant articles using a search engine (e.g. Bing or Google Search), which are then split into chunks. We then compute the embeddings of each chunk, rank them by cosine similarity with respect to the embedding of the query, and put the most relevant chunks into a prompt to generate the final answer, while also keeping track of the sources.

Ask Trending Questions
Let’s start this repo by seeing an example. The following piece must be familiar by now. It uses the OpenAI GPT-3.5-turbo model to create an assistant to answer questions. We will ask the model to name the latest Fast & Furious movie, released recently. Therefore, the model couldn’t have seen the answer during the training. Remember to install the required packages with the following command: pip install langchain deeplake openai tiktoken.
```

```
The latest Fast and Furious movie is Fast & Furious 9, which is set to be released in May 2021.

```

The response shows that the model references the previous movie title as the answer. This is because the new movie (10th sequel) has yet to be released in its fictional universe! Let’s fix the problem.

## Google API
Before we start, let’s set up the API Key and a custom search engine. If you don’t have the keys from the previous repo, head to the [Google Cloud console](https://console.cloud.google.com/apis/credentials) and generate the key by pressing the CREATE CREDENTIALS buttons from the top and choosing API KEY. Then, head to the [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/create) dashboard and remember to select the “Search the entire web” option. The Search engine ID will be visible in the details. You might also need to enable the “Custom Search API” service under the Enable APIs and services. (You will receive the instruction from API if required) Now we can set the environment variables for both Google and OpenAI APIs.

```
import os

os.environ["GOOGLE_CSE_ID"] = "<Custom_Search_Engine_ID>"
os.environ["GOOGLE_API_KEY"] = "<Google_API_Key>"
os.environ["OPENAI_API_KEY"] = "<OpenAI_Key>"
```

## Get Search Results
This section uses LangChain’s GoogleSearchAPIWrapper class to receive search results. It works in combination with the Tool class that presents the utilities for agents to help them interact with the outside world. In this case, creating a tool out of any function, like top_n_results is possible. The API will return the page’s title, URL, and a short description.

```
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
TOP_N_RESULTS = 10

def top_n_results(query):
    return search.results(query, TOP_N_RESULTS)

tool = Tool(
    name = "Google Search",
    description="Search Google for recent results.",
    func=top_n_results
)

query = "What is the latest fast and furious movie?"

results = tool.run(query)

for result in results:
    print(result["title"])
    print(result["link"])
    print(result["snippet"])
    print("-"*50)
```

```
Fast & Furious movies in order | chronological and release order ...
https://www.radiotimes.com/movies/fast-and-furious-order/
Mar 22, 2023 ... Fast & Furious Presents: Hobbs & Shaw (2019); F9 (2021); Fast and Furious 10 (2023). Tokyo Drift also marks the first appearance of Han Lue, a ...
--------------------------------------------------
FAST X | Official Trailer 2 - YouTube
https://www.youtube.com/watch?v=aOb15GVFZxU
Apr 19, 2023 ... Fast X, the tenth film in the Fast & Furious Saga, launches the final ... witnessed it all and has spent the last 12 years masterminding a ...
--------------------------------------------------
Fast & Furious 10: Release date, cast, plot and latest news on Fast X
https://www.radiotimes.com/movies/fast-and-furious-10-release-date/
Apr 17, 2023 ... Fast X is out in cinemas on 19th May 2023 – find out how to rewatch all the Fast & Furious movies in order, and read our Fast & Furious 9 review ...
--------------------------------------------------
Fast & Furious - Wikipedia
https://en.wikipedia.org/wiki/Fast_%26_Furious
The main films are known as The Fast Saga. Universal expanded the series to include the spin-off film Fast & Furious Presents: Hobbs & Shaw (2019), ...
--------------------------------------------------
How many 'Fast & Furious' movies are there? Here's the list in order.
https://www.usatoday.com/story/entertainment/movies/2022/07/29/fast-and-furious-movies-order-of-release/10062943002/
Jul 29, 2022 ... There are currently nine films in the main "Fast and Furious" franchise, with the 10th, "Fast X," set to release on May 19, 2023. There are ...
--------------------------------------------------
How to Watch Fast and Furious Movies in Chronological Order - IGN
https://www.ign.com/articles/fast-and-furious-movies-in-order
Apr 6, 2023 ... Looking to go on a Fast and Furious binge before the next movie comes out? ... This is the last Fast film with Paul Walker's Brian O'Conner, ...
--------------------------------------------------
'Fast and Furious 10': Everything We Know So Far
https://www.usmagazine.com/entertainment/pictures/fast-and-furious-10-everything-we-know-so-far/
7 days ago ... Fast X will be the second-to-last film in the blockbuster franchise, and Dominic Toretto's next adventure is set to be one of the biggest so far ...
--------------------------------------------------
Latest 'Fast & Furious' Movie Leads Weekend Box Office - WSJ
https://www.wsj.com/articles/latest-fast-furious-movie-leads-weekend-box-office-11624815451
Jun 27, 2021 ... “F9,” however, offers the clearest test yet on the post-pandemic habits of moviegoers. The movie is the most-anticipated title released since ...
--------------------------------------------------
Fast & Furious Movies In Order: How to Watch Fast Saga ...
https://editorial.rottentomatoes.com/guide/fast-furious-movies-in-order/
After that, hop to franchise best Furious 7. Follow it up with The Fate of the Furious and spin-off Hobbs & Shaw and then the latest: F9 and Fast X. See below ...
--------------------------------------------------
The looming release of the latest "Fast and Furious" movie heightens ...
https://www.cbsnews.com/losangeles/news/looming-release-of-latest-fast-and-furious-movie-heightens-concerns-over-street-racing-takeovers/
17 hours ago ... With the latest installment of the "Fast and Furious" franchise set to hit theaters this weekend, local law enforcement is banding together ...
--------------------------------------------------
```

Now, we use the results variable’s link key to download and parse the contents. The newspaper library takes care of everything. However, it might be unable to capture some contents in certain situations, like anti-bot mechanisms or having a file as a result.

```

import newspaper

pages_content = []

for result in results:
	try:
		article = newspaper.Article(result["link"])
		article.download()
		article.parse()
		if len(article.text) > 0:
			pages_content.append({ "url": result["link"], "text": article.text })
	except:
		continue
```

## Process the Search Results
We now have the top 10 results from the Google search. (Honestly, who looks at Google’s second page?) However, it is not efficient to pass all the contents to the model because of the following reasons:

- The model’s context length is limited.
- It will significantly increase the cost if we process all the search results.
- In almost all cases, they share similar pieces of information.

So, let’s find the most relevant results, 

Incorporating the LLMs embedding generation capability will enable us to find contextually similar content. It means converting the text to a high-dimensionality tensor that captures meaning. The cosine similarity function can find the closest article with respect to the user’s question.

It starts by splitting the texts using the RecursiveCharacterTextSplitter class to ensure the content lengths are inside the model’s input length. The Document class will create a data structure from each chunk that enables saving metadata like URL as the source. The model can later use this data to know the content’s location.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

docs = []
for d in pages_content:
	chunks = text_splitter.split_text(d["text"])
	for chunk in chunks:
		new_doc = Document(page_content=chunk, metadata={ "source": d["url"] })
		docs.append(new_doc)
```

The subsequent step involves utilizing the OpenAI API's OpenAIEmbeddings class, specifically the .embed_documents() method for search results and the .embed_query() method for the user's question, to generate embeddings.

```
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

docs_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
query_embedding = embeddings.embed_query(query)
```

Lastly, the get_top_k_indices function accepts the content and query embedding vectors and returns the index of top K candidates with the highest cosine similarities to the user's request. Later, we use the indexes to retrieve the best-fit documents.

```
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
  # convert the lists of vectors to numpy arrays
  list_of_doc_vectors = np.array(list_of_doc_vectors)
  query_vector = np.array(query_vector)

  # compute cosine similarities
  similarities = cosine_similarity(query_vector.reshape(1, -1), list_of_doc_vectors).flatten()

  # sort the vectors based on cosine similarity
  sorted_indices = np.argsort(similarities)[::-1]

  # retrieve the top K indices from the sorted list
  top_k_indices = sorted_indices[:top_k]

  return top_k_indices

top_k = 2
best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
best_k_documents = [doc for i, doc in enumerate(docs) if i in best_indexes]
```

## Chain with Source
Finally, we used the selected articles in our prompt (using the stuff method) to assist the model in finding the correct answer. LangChain provides the load_qa_with_sources_chain() chain, which is designed to accept a list of input_documents as a source of information and a question argument which is the user’s question. The final part involves preprocessing the model’s response to extract its answer and the sources it utilized.

```
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

response = chain({"input_documents": best_k_documents, "question": query}, return_only_outputs=True)

response_text, response_sources = response["output_text"].split("SOURCES:")
response_text = response_text.strip()
response_sources = response_sources.strip()

print(f"Answer: {response_text}")
print(f"Sources: {response_sources}")
```

```
Answer: The latest Fast and Furious movie is Fast X, scheduled for release on May 19, 2023.
Sources: https://www.radiotimes.com/movies/fast-and-furious-10-release-date/, https://en.wikipedia.org/wiki/Fast_%26_Furious
```

The use of search results helped the model find the correct answer, even though it never saw it before during the training stage. The question and answering chain with source also provides information regarding the sources utilized by the model to derive the answer.
