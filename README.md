# reverse-rag

Portable Training Data Generation for Supervised Fine-tuning: A Reverse RAG approach!

Details in the blog: [https://medium.com/google-cloud/portable-training-data-generation-for-supervised-fine-tuning-a-reverse-rag-approach-ee05a9b26114](url)

**Reverse-RAG!**

Collect raw information (can be just a wall of text or a lump of data) that can source your datasets for training.
Engineer a prompt using Gemini 1.5 Flash 001 model with few examples to generate as many question and answers.
Configure the output to be in JSONL format.
Export it to Cloud Storage.
Kickstart your Gemini 1.0 Pro Fine Tuning step!
Let’s try it!

**Step 1:**

Here is my raw information: Big blog about something nice

**Step 2:**

Here is my prompt: prompt.txt in this repo

**Step 3:**

Here is the code to configure the context, parameters and the prompt: Reverse Rag.ipynb file in the 'Reverse RAG' folder this repo

The response I received: response_sample.txt in this repo

**Step 4:**

Export this as a JSONL file in Cloud Storage.

**Step 5:**

Fine Tune! You can do this from Vertex AI Create a tuned model page on the Google Cloud console or even programmatically. Here is the notebook that you can run from Colab research or Colab Enterprise on Vertex AI:

**TEST USING THE NEW MODEL TUNED:**

Blog Fine Tuning Notebook in this repo has the code for step 5 and testing using the fine tuned and base model.

**Conclusion:
**
Using an arbiter method to reverse RAG a summarized and nimble dataset from vast amounts of raw data applies to a lot of use cases:

Prompt Engineering: By generating diverse and relevant questions based on a knowledge base, this approach can aid in crafting more effective prompts for LLMs, improving their performance across various tasks.
Few-Shot Prompting: It can be used to generate synthetic examples for few-shot prompting, providing LLMs with additional context and guidance for specific tasks.
Evaluation of Agentic Applications: The generated Q&A pairs can serve as a benchmark for evaluating the performance and capabilities of agentic applications, ensuring they align with the desired knowledge and behavior.
RAG & Fine Tuning: Enhancing the performance of RAG applications by grounding it to a source of truth while keeping the input token length concise.
