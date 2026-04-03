# Dataset Recommender

A machine learning pipeline designed to retrieve, score, and rank open source datasets from multiple repositories simultaneously. This tool helps data scientists find optimal data for specific tasks like classification, anomaly detection, and regression.

## System Architecture

The application uses an asynchronous retrieval pipeline paired with a multi-stage ranking algorithm.

* **Asynchronous Retrieval:** Fetches maximum candidate datasets from Kaggle, Hugging Face, and Data.gov concurrently.
* **Intent Parsing:** Uses regular expressions to extract hard constraints like class balance, dataset size, and licensing requirements directly from the user query.
* **Semantic Search:** Converts queries and dataset metadata into vector embeddings using the all-MiniLM-L6-v2 model. It builds a local FAISS index to calculate cosine similarity and normalize the vector distances.
* **Heuristic Metadata Scoring:** Evaluates dataset metadata including update recency, file size, license openness, and file format suitability.
* **LLM Evaluation:** Passes the top filtered candidates to the Qwen2.5-7B-Instruct language model via the Hugging Face Inference API. The model evaluates contextual relevance and generates a custom justification for each dataset.

## Scoring Algorithm

The final suitability score is calculated using a dynamic weighted ensemble. The system computes the relevance score by applying intent-based weight boosts to base configuration weights. 

The final calculation follows this mathematical structure:

$$\text{Relevance} = \sum_{i=1}^{n} (w_i \times s_i)$$

Where:
$w_i$: Represents the dynamically resolved weight for a specific dimension.
$s_i$: Represents the normalized score for that dimension.

## Technology Stack

* **Backend:** Python, asyncio, aiohttp
* **Machine Learning:** Sentence Transformers, FAISS, Hugging Face Hub
* **User Interface:** Gradio
* **APIs:** Kaggle API, Data.gov CKAN API
* **Containerization:** Podman

## Environment Configuration

Create a .env file in the root directory and add the required API tokens.

```text
HF_TOKEN=hf_your_huggingface_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
``` 

## Podman Deployment
Running the application via Podman ensures complete environment isolation without requiring heavy desktop applications.

1. Ensure your Gradio launch script in app.py is set to broadcast to all network interfaces:

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

2. Build the container image:

```bash
podman build -t dataset-recommender .
```

3. Start the container in detached mode with auto-cleanup and explicit IPv4 binding to bypass Windows routing conflicts:

```bash
podman run -d --rm -p 127.0.0.1:7861:7860 --env-file .env dataset-recommender
```

4. Open a web browser and navigate to:
**http://127.0.0.1:7861**

5. To stop the application safely, locate your container ID and issue the stop command:

```bash
podman ps
podman stop <container_id>
```

## Local Installation
1. Initialize a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment for Windows:

```bash
.\venv\Scripts\activate
```

3. Activate the environment for Mac or Linux:

```bash
source venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Ensure your Gradio launch script in app.py is set to local binding:

```python
demo.launch()
```

6. Launch the Gradio server:

```bash
python app.py
```
