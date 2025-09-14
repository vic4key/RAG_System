# RAG System

A flexible and extensible Retrieval-Augmented Generation (RAG) system that supports multiple popular embedding providers, caching, adapters for various data types, and seamless integration with vector databases (ChromaDB).

---

## 🚀 Key Features

- **Pluggable Embedding Providers**
  - Supports: OpenAI, Azure OpenAI, Cohere, Google PaLM (OpenAI-compatible), SentenceTransformers, HuggingFace Transformers.
  - Easily extendable to add new providers.

- **Flexible Data Adapters**
  - Easily add adapters for various data sources: JSON, PDF, Text, etc.

- **Vector Database Integration**
  - Uses ChromaDB (local or persistent).
  - Automatically checks, creates, and manages collections.

- **Query Caching**
  - Supports caching of query results for speed and cost efficiency.
  - Can be enabled/disabled via environment variable.

- **Flexible Configuration via `.env`**
  - Select provider, model, endpoint, API key, etc. through environment variables.

- **Easy Integration and Extension**
  - Clean, maintainable codebase.
  - Includes examples and unit tests.

## 💳 License
- 📰 Released under the [MIT](LICENSE) license
- ©️ Copyright © Vic P. & Vibe Coding ❤️

## 🧩 Usage

💻 **Installation**

```bash
$ cd to/your/local/repo
$ git submodule add https://github.com/vic4key/RAG_System.git
$ pip install -r RAG_System/requirements.txt
$ python -m RAG_System.examples.rag.test # make sure it works
$ cp RAG_System/.env.example .env
```

⚙️ **Configuration**

Remember to update the `.env` file (see `.env` for all available options)

🚀 **Coding**

<details>
<summary>Click here to view more ...</summary>

```python
from RAG_System import RAG_System
from RAG_System.examples.sample_faqs.sample_faqs_adapter import SampleFAQsAdapter

from dotenv import load_dotenv
load_dotenv()

similarity_threshold = 0.4

adapter = SampleFAQsAdapter()

rag = RAG_System()
rag.setup_from_adapter(
    adapter=adapter,
    adapter_params={
        "file_path": "sample_faqs.json",
    },
    collection_name="sample_faqs",
    persist_directory=f"chroma_db/sample_faqs",
    recreate=False,
)

questions = [
    "Làm sao để chỉnh sửa hoặc xóa chi tiêu?",
    "Xóa chi tiêu?",
    "How to connect my bank account?",
]

for question in questions:
    if result_items := rag.query(question):
        if best_match := result_items[0]:
            print(f"Found {len(result_items)} result items")
            print(f"Best match:")
            print(f"   Question: {best_match.metadata.get('question', '')}")
            print(f"   Answer:   {best_match.metadata.get('answer', '')}")
            print(f"   Distance: {best_match.metadata.get('distance', 1.0)} ({similarity_threshold = })")
        print(f"\nRetrieval:\n```\n{result_items}\n```")
        print(f"\nGeneration:")
        for chunk in rag.generate(question, result_items, stream=True):
            print(chunk, end="", flush=True)
    else:
        print(f"Not found")
```
</details>

## 🧩 Development

💻 **Installation**

```bash
$ git clone https://github.com/vic4key/RAG_System.git
$ pip install -r RAG_System/requirements.txt
$ python -m RAG_System.examples.rag.test
```

📚 **Extending & Customization**

- **Add a new provider:**  
  Create a new provider class inheriting from `BaseEmbeddingProvider` in `core/providers/` and register it in the factory.

- **Add a new data adapter:**  
  Create a new adapter class inheriting from `RAG_Adapter` in `core/adapters/` and register it in the factory.

## 🧪 Examples

The [examples](examples) folder contains usage examples and unit tests for each data type, caching, etc.

## 📬 Contact

Feel free to contact via [Twitter](https://twitter.com/vic4key) / [Gmail](mailto:vic4key@gmail.com) / [Blog](https://blog.vic.onl/) / [Website](https://vic.onl/)