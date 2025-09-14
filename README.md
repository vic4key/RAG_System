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
pip install -r requirements.txt
```

⚙️ **Configuration**

Remember to configure `.env`. See `.env.example` for all available options.

🚀 **Coding**

```python
from rag_system import RAG_System

rag = RAG_System()
results = rag.query("I want to ask about the refund policy")
```

📚 **Extending & Customization**

- **Add a new provider:**  
  Create a class inheriting from `BaseEmbeddingProvider` in `core/embedding_providers.py` and register it in the factory.
- **Add a new data adapter:**  
  Create a new adapter in `core/adapters/` and register it with the system.

## 🧪 Examples

The [examples](examples) folder contains usage examples and unit tests for each data type, caching, etc.

## 📬 Contact

Feel free to contact via [Twitter](https://twitter.com/vic4key) / [Gmail](mailto:vic4key@gmail.com) / [Blog](https://blog.vic.onl/) / [Website](https://vic.onl/)