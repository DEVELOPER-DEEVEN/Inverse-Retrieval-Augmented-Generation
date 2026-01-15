# **Supervised Fine-tuning: A Reverse RAG approach!**

*I just made up Reverse-RAG. So judge fast but thank me later.*

**Fine tuning makes LLMs more task-specific**

In the rapidly evolving landscape of large language models (LLMs), fine-tuning is critical in adapting these powerful tools to specific tasks and domains.

**However**

Fine tuning LLMs typically requires extensive datasets comprising instructions and corresponding desired responses, hence can be a resource-intensive and time-consuming process.

Additionally, organizations often lack the luxury of readily available training data for novel use cases. The laborious task of preparing sample datasets further worsens this challenge, demanding substantial resources and financial investment.

**Once upon a time**

Before LLMs, it was just wishful thinking to imagine an automated training dataset generation process. Then came RAGs that made it simpler to imagine retrieval augmented generation — that is to derive “answers” from knowledge that your organization has.

**The Hidden Challenge of RAG**

In the world of Retrieval Augmented Generation (RAG) and generative AI-powered chat applications, the most daunting hurdle often lies in the unknown: identifying the right questions and extracting pertinent information from an available knowledge base. New organizations frequently grapple with the absence of clear boundaries and planned conversational structures, leading to ambiguity in data preparation for model fine-tuning.

**Enter Training Dataset Automation: A Reverse RAG Approach**

This is where automation with generative AI emerges as a game-changer. By automating the creation of training datasets, developers can significantly streamline the fine-tuning process for large language models (LLMs).

Just like we use an arbiter model to evaluate another LLM, this blog will introduce you to an automated approach to fine-tune Gemini 1.0 Pro using Gemini 1.5 Flash to simplify training dataset creation and to accelerate fine tuning. **In essence I am just going to generate question & answer sets in a specific template using an available knowledge base.**

> ***Demystifying the jargons***
>
> ***“Arbiter Method”: ****The essence of the “Arbiter Method” lies in employing a second LLM, in this case, Gemini 1.5 Flash, to act as a discerning judge or evaluator during the training dataset generation process. Think of it as a quality control mechanism. In essence, we feed the raw information to Gemini 1.5 Flash and instruct it to generate potential questions and answers based on that content. The “arbiter” model then scrutinizes these generated Q&A pairs, filtering out irrelevant or inaccurate ones, ensuring that only the most pertinent and contextually sound information makes its way into the final training dataset.*
>
> ***“Reverse-RAG:” ****Reverse-RAG (a terminology I made up to represent a shift in the approach to training dataset generation). Why “reverse”? Because usually RAG refers to the process of using knowledge base to retrieve answers, and here we use the answers to postulate questions, which we then use as the training dataset or context.*
>
> *Note: We are not generating arbitrary datasets using the generative LLMs but using raw data as the source of truth to generate datasets.*

**Use Case**

Let’s say I want to build a chat application that uses generative AI to answer specific questions for a learning portal . If I were to use a Generative AI model as it is, without case-specific information, I will end up getting open ended responses and non-deterministic answers that may or may not be directly derived from the artifacts in our repository.

Example: Try this prompt out in any LLM:

> How can a Vertex AI agent app be integrated to a web application?

Your response could look like anything. I got this but this is clearly because my prompt is very vague and open-ended:
<img width="1338" height="1049" alt="Screenshot 2025-10-14 at 11 28 27 AM" src="https://github.com/user-attachments/assets/390827dc-3149-4e79-9412-6e144a475c29" />


But I am looking for a response as simple as this:

> Using the Dialogflow CX library and the `detectIntent` method. This allows web applications to interact with the Dialogflow CX API and access agent responses.

**Solution**

**Fine-tuning!!!** Let’s say I am going to fine-tune Gemini 1.5 Pro to get the desired result. To generate the required training datasets, I am going to use what I just made up:

**Reverse-RAG!**
1. Collect raw information (can be just a wall of text or a lump of data) that can source your datasets for training.
2. Engineer a prompt using Gemini 1.5 Flash 001 model with few examples to generate as many question and answers.
3. Configure the output to be in JSONL format.
4. Export it to Cloud Storage.
5. Kickstart your Gemini 1.0 Pro Fine Tuning step!

**Let’s try it!**


**Step 1:**

Here is my prompt:

```
Below is my blog about building agentic app for restaurant review chat assistant. 
I want you to read the blog and generate a JSONL text in response with all possible questions and answers from the blog. 
Remember you cannot make up your own content, the facts should be strictly derived from the blog content. 
Also adhere to the response structure mentioned in the example in the end of this prompt.

Blog:
<< Insert multimodal raw data here >>
The blog has ended.

Example for what your JSONL response should look like:
{"messages": [{"role": "user", "content": "What is the tech stack used in this use case?\n"}, {"role": "model", "content":"Tech stack
The front end of the application is built with Angular. The app is hosted on Firebase Hosting.

The rest of the technologies are described below:

MongoDB Atlas — a fully managed cloud database service that handles deployment, scaling, and operations for MongoDB

MongoDB Atlas Data API — a REST-like interface to interact with MongoDB data, enabling easy integration with various applications and services
MongoDB Atlas Functions — serverless JavaScript functions running in MongoDB Atlas
MongoDB Atlas Triggers — Event-driven triggers within MongoDB Atlas that trigger Atlas functions based on database changes
Analyze sentiment Atlas trigger
Summarize reviews Atlas trigger
Analyze media Atlas trigger
Google Cloud

Vertex AI Gemini API — the gemini-1.0-pro and gemini-1.0-pro-vision models
Google Cloud Functions
Google Cloud Storage
Firebase Hosting"}]}
{"messages": [{"role": "user", "content": "What is Sentiment Chef and what is its purpose?\n"}, {"role": "model", "content":"Sentiment Chef is a web app that uses MongoDB Atlas Triggers to capture new restaurant reviews and send them to Google Cloud Functions. These functions leverage the power of Gemini, a cutting-edge large language model (LLM), to analyze sentiment (positive, neutral, negative) and generate concise summaries of the text. We also used Gemini to extract sentiment and tags from images and videos attached to reviews. The goal of Sentiment Chef is to quickly assess customer sentiment and provide a snapshot of the key takeaways from reviews."}]}
```

**Step 2:**

Here is the code to configure the context, parameters and the prompt:

```
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate():
  vertexai.init(project="<<>>", location="us-central1")
  model = GenerativeModel(
    "gemini-1.5-flash-001",
  )
  responses = model.generate_content(
      [text1],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )

  for response in responses:
    print(response.text, end="")

text1 = """ ADD THE PROMPT FROM ABOVE STEP HERE  """

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

generate()

```

Below is the response I received:

```
{"messages": [{"role": "user", "content": "What problem does Sentiment Chef aim to solve?"}, {"role": "model", "content": "Sentiment Chef aims to solve the problem of extracting meaning from large volumes of customer reviews. These reviews are often buried in paragraphs of text or hidden within videos and photos, making it challenging to understand the overall sentiment and key takeaways."}]}
{"messages": [{"role": "user", "content": "What is the initial solution offered by Sentiment Chef?"}, {"role": "model", "content": "Sentiment Chef initially uses MongoDB Atlas Triggers to capture new restaurant reviews and send them to Google Cloud Functions. These functions leverage Gemini, a large language model, to analyze sentiment (positive, neutral, negative) and generate concise summaries of the text. Gemini also extracts sentiment and tags from attached images and videos."}]}
{"messages": [{"role": "user", "content": "How does Sentiment Chef evolve from simple analysis to actionable insights?"}, {"role": "model", "content": "Sentiment Chef evolves by integrating restaurant review data with other relevant information like restaurant details, cuisine, and location using MongoDB Atlas Data API. It builds an analytics engine using BigQuery and its machine learning capabilities to process this expanded dataset, create embeddings for semantic search, and generate nuanced responses to user queries. This enables the creation of a conversational agent using Vertex AI Agent console, providing users with personalized restaurant recommendations."}]}
{"messages": [{"role": "user", "content": "What are the capabilities of the evolved Sentiment Chef as a conversational agent?"}, {"role": "model", "content": "The evolved Sentiment Chef is a sophisticated chatbot that understands natural language and can answer questions like: \n\n* What are the best Italian restaurants near me with outdoor seating?\n* I’m looking for a vegan-friendly restaurant with a lively atmosphere.\n* Can you recommend a quiet place for a romantic dinner?\n\nIt uses the data stored in MongoDB Atlas and insights generated by the BigQuery analytics engine to provide accurate and helpful responses."}]}
{"messages": [{"role": "user", "content": "What are the two main components of the final Sentiment Chef project?"}, {"role": "model", "content": "The final Sentiment Chef project has two main components: \n\n1. A web application for submitting and analyzing restaurant reviews.\n2. An intelligent chatbot agent for interpreting the reviews."}]}
{"messages": [{"role": "user", "content": "How does Sentiment Chef analyze media (images and videos) attached to reviews?"}, {"role": "model", "content": "Media files are uploaded to a Google Cloud Storage bucket. Then, a Google Cloud function uses Gemini to extract metadata (tags, sentiment, descriptions) from the uploaded files. This enriched metadata is stored in MongoDB Atlas along with the image URLs, contributing to the overall sentiment of the review."}]}
{"messages": [{"role": "user", "content": "How does Sentiment Chef handle review submission and analysis?"}, {"role": "model", "content": "Submitted reviews are stored in a MongoDB Atlas collection called `raw_reviews`. An Atlas trigger forwards this data to a Google Cloud function. The function uses Vertex AI Gemini to analyze the review text and returns the sentiment data to the Atlas function. This enriched review is then stored in the `processed_reviews` collection. The web app actively monitors for new processed reviews and displays them in the UI."}]}
{"messages": [{"role": "user", "content": "What is the purpose of the Restaurant Agent chat assistant?"}, {"role": "model", "content": "The Restaurant Agent is a chat assistant that helps users find restaurants that match their needs by answering questions based on review data. It taps into detailed data in MongoDB Atlas and insights from a BigQuery analytics engine to provide accurate and useful responses."}]}
{"messages": [{"role": "user", "content": "How is data ingested into the BigQuery analytics engine?"}, {"role": "model", "content": "Data can be ingested into BigQuery using MongoDB Atlas Data API or MongoDB to BigQuery dataflow templates. In this case, the data is pre-processed and converted to JSONL format before being ingested into BigQuery tables."}]}
{"messages": [{"role": "user", "content": "How are remote models used in the analytics engine?"}, {"role": "model", "content": "Remote models allow BigQuery to interact with Vertex AI models. To use them, an external connection is created in BigQuery, allowing access to models like `textembedding-gecko` for embedding generation."}]}
{"messages": [{"role": "user", "content": "What is the purpose of the analytics engine in the Restaurant Agent?"}, {"role": "model", "content": "The analytics engine transforms raw data into actionable insights. It extracts context, creates embeddings, and drives the search for relevant information. This engine is built in BigQuery."}]}
{"messages": [{"role": "user", "content": "How does the analytics engine summarize restaurant data?"}, {"role": "model", "content": "The analytics engine summarizes restaurant data by combining restaurant details, reviews, and grades into a comprehensive description. This summary is then converted into vector embeddings using the `textembedding-gecko` model and stored in a separate table."}]}
{"messages": [{"role": "user", "content": "Explain the similarity search process for user queries in the analytics engine."}, {"role": "model", "content": "The user's search text is converted into embeddings and matched against the embeddings in the summary table using Vector Search (Cosine Distance Method). Gemini 1.5 Pro then validates the relevance of the matches by assigning a percentage or score. The top five most relevant matches are returned to the user as a chat response."}]}
{"messages": [{"role": "user", "content": "How does the Cloud Function process user input for the Restaurant Agent?"}, {"role": "model", "content": "A Cloud Function receives the user's search text as a JSON request. It then invokes the BigQuery SQL query to perform the similarity search and returns the top five relevant matches as a JSON array string. This function is exposed as a REST API endpoint."}]}
{"messages": [{"role": "user", "content": "What is Vertex AI Agent Builder and how is it used in this project?"}, {"role": "model", "content": "Vertex AI Agent Builder provides a no-code console for building generative AI agents using natural language instructions and conversational examples. In this project, Agent Builder is used to connect with the Cloud Function endpoint and enable the Restaurant Agent to access the knowledge base and respond to customer queries intelligently."}]}
{"messages": [{"role": "user", "content": "What are the instructions provided to the Restaurant Agent in Agent Builder?"}, {"role": "model", "content": "The instructions for the Restaurant Agent in Agent Builder include:\n\n* Greeting the user and asking how to help.\n* Using personal details if provided.\n* Summarizing the user's request and asking for confirmation.\n* Seeking clarifying details if the request is broad.\n* Narrowing down the request with specific details if necessary.\n* Summarizing the final understanding of the request.\n* Using the 'Review Tool' to find relevant restaurants.\n* Gracefully handling requests outside the scope of the tool.\n* Avoiding providing information beyond the tool's knowledge.\n* Asking if the user has any other queries.\n* Thanking the user and saying goodbye."}]}
{"messages": [{"role": "user", "content": "How can the Restaurant Agent be integrated into a web application?"}, {"role": "model", "content": "The Restaurant Agent can be integrated into a web application using the Dialogflow CX library and the `detectIntent` method. This allows the web application to interact with the Dialogflow CX API and access the Restaurant Agent's responses."}]}
{"messages": [{"role": "user", "content": "What are some potential future improvements for Sentiment Chef?"}, {"role": "model", "content": "Potential future improvements for Sentiment Chef include:\n\n* Multimodal prompts: Allowing users to search with text and images (e.g., \"Find me a restaurant that looks like this picture\").\n* Enhanced personalization: Tailoring recommendations based on individual user preferences and past dining experiences.\n* Integration with reservation systems: Enabling users to book tables directly through the chat interface."}]}
{"messages": [{"role": "user", "content": "What is the overall conclusion of this project?"}, {"role": "model", "content": "Building a smart restaurant review summary and search agent with MongoDB, BigQuery, Gemini 1.5 Pro, and Agent Builder was a challenging but satisfying project. By leveraging these technologies, a conversational agent was created that can provide users with comprehensive restaurant information. This agent enhances the customer experience and showcases the potential of AI in revolutionizing the hospitality industry."}]}
```

**Step 3:**

Export this as a JSONL file in [__Cloud Storage__](https://console.cloud.google.com/storage).

**Step 4:**

**Fine Tune!** You can do this from Vertex AI [*__Create a tuned model__*](https://console.cloud.google.com/vertex-ai/generative/language/create-tuning) page on the Google Cloud console or even programmatically. Here is the notebook that you can run from [__Colab__](https://colab.research.google.com/) research or [__Colab Enterprise on Vertex AI__](https://console.cloud.google.com/vertex-ai/colab/notebooks):

Once you run the Tuning Job step, you will encounter the following stages:

Tuning Job kickoff:

Data Distribution:

<img width="710" height="561" alt="Screenshot 2025-10-14 at 11 17 50 AM" src="https://github.com/user-attachments/assets/10d33ff8-a349-4de1-ab87-3f0aadda8eec" />


It will take several minutes for the tuning job to be completed.

**Let’s test the newly tuned and deployed model**

Run the response against the validation question we started off this use case with,

```
"How can the Vertex AI agent app be integrated to a web application?"
```

using the newly tuned and deployed model. Here is the code for it:

```
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview import tuning
from vertexai.preview.tuning import sft

sft_tuning_job = sft.SupervisedTuningJob("projects/YOUR_PROJECT_NUMBER/locations/us-central1/tuningJobs/YOUR_TUNING_JOB_ID")
tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


def generate():
  vertexai.init(project=project_id, location="us-central1")
  text1 = question + "Here is the source of truth: " + information
  responses = tuned_model.generate_content(
      [text1],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )

  for response in responses:
    print(response.text, end="")


generate()
```

Note that the variables “question” and “information” contain the validation question and the re-RAG’ed information in the JSONL format respectively.

Here is the response I got:

```
## Integration: Use the Dialogflow CX library and its `detectIntent` method to integrate the Vertex AI agent app into your web application.
```

Seems pretty good to me.

**Let’s test the same prompt against the foundational LLM without tuning and with raw data**

In this test, we will replace the information with the raw data (before re-RAG process):

```
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


def generate():
  vertexai.init(project=project_id, location="us-central1")
  model = GenerativeModel(
    "gemini-1.0-pro-002",
  )
  text1 = question + "Here is the source of truth: " + raw_content
  responses = model.generate_content(
      [text1],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )

  for response in responses:
    print(response.text, end="")


generate()
```

Note that the variables “question” and “information” contain the validation question and the blog in the raw text format respectively.

Here is the response:

```
## How to integrate Vertex AI agent to a web application

There are two main approaches to integrate the Vertex AI agent to a web application, based on the information provided:

**1. Using the Agent Builder Agent app:**

* This option involves leveraging the Agent Builder platform to create an agent that can be seamlessly integrated with your web application. 
* The agent can be configured to respond to user queries, search the restaurant database, and provide summarized information about restaurants.
* The integration can be achieved using the Dialogflow CX API and the detectIntent method, as described in the document.

**2. Using the OpenAPI spec in YAML format:**

* This option involves directly calling the Cloud Function endpoint using the OpenAPI spec provided in the document. 
* The OpenAPI spec defines the request and response format for the Cloud Function, allowing your web application to interact with the agent.
* This approach offers more flexibility in terms of how you can integrate the agent into your web application.

Both approaches have their own advantages and disadvantages. The Agent Builder Agent app offers a more user-friendly interface and requires less coding, while the OpenAPI spec approach provides more flexibility and control. Ultimately, the best approach will depend on your specific needs and requirements.

Here is a one-line summary of how to integrate the Vertex AI agent to a web application:

**Use either the Agent Builder Agent app or the OpenAPI spec in YAML format to connect your web application to the Cloud Function endpoint, which will then interact with the BigQuery analytics engine to provide restaurant information based on user queries.**
```

As you can observe, without improving prompt contexts with the re-RAG information and without fine tuning, we have a pretty long and open ended response. Even though it may be factually correct, it may not adhere in structure and relevance to what we want to show to the user as the answer that is grounded in structure and content to our preferences.

> *Why did I use the JSONL format Q&A training dataset as context in addition to fine tuning the model?*
>
> *I have very limited training data for my sample use case. I can use a lot more information from raw data out there in order to generate more training data or choose to ground the fine tuned model once again with the JSONL dataset. This is optional, but it helps improve the accuracy and determinism in the responses. Of course, when we are productionizing applications using generative AI, we need to consider context window, number of tokens and ensure the response and the summaries are concise and laser focused.*

**Advanced: Bayesian Hyperparameter Tuning**

To ensure the generated Q&A dataset is of the highest quality, we can employ Bayesian Optimization to tune the `temperature` and `top_p` parameters of the Gemini 1.5 Flash model. We have implemented a script `bayesian_optimizer.py` that uses [Optuna](https://optuna.org/) to automate this process.

The optimizer works by:
1.  **Generating** a dataset with specific hyperparameters.
2.  **Evaluating** the quality of the dataset using **Gemini 1.5 Pro** as a "Judge" (scoring it on factual consistency, format validity, and relevance).
3.  **Iterating** to find the hyperparameters that maximize this quality score.

To run the optimizer:
1.  Install dependencies: `pip install optuna google-cloud-aiplatform`
2.  Update `PROJECT_ID` in `bayesian_optimizer.py`.
3.  Run: `python bayesian_optimizer.py`

The script will output the best parameters and save the optimized dataset to `best_dataset.jsonl`, ensuring your fine-tuning uses the most robust data possible.

**Conclusion:**

Using an arbiter method to reverse RAG a summarized and nimble dataset from vast amounts of raw data applies to a lot of use cases:
- [**__Prompt__**](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts#fundamentals)** Engineering:** By generating diverse and relevant questions based on a knowledge base, this approach can aid in crafting more effective prompts for LLMs, improving their performance across various tasks.
- **Few-Shot Prompting:** It can be used to generate synthetic examples for few-shot prompting, providing LLMs with additional context and guidance for specific tasks.
- [**__Evaluation__**](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)** of Agentic Applications:** The generated Q&A pairs can serve as a benchmark for evaluating the performance and capabilities of agentic applications, ensuring they align with the desired knowledge and behavior.
- **RAG & Fine Tuning:** Enhancing the performance of RAG applications by [__grounding__](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview#ground-public) it to a source of truth while keeping the input token length concise.
