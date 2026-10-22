# DeepSeek LLM for Structured Release Extraction

Subreddit titles lack a standard schema and include irregular delimiters, brackets, release dates, and retailer tags that make regex extraction brittle. We decided to use DeepSeek (`deepseek-chat`) via PydanticAI with structured schema validation to normalize unstructured post titles into validated release models. This yields high semantic extraction accuracy while keeping cost and inference latency low.
