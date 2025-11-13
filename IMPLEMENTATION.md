# What we implemented
We implemented the functionality to convert unstructured information into vectors using an embedding model. We also deployed the model on a personal computer through Hugging Face. In addition, we implemented the ability to align vectors with the positions of the original text via an index file.

# How we implemented
We used pair programming. Because our code components are highly interconnected, we had to work closely together, reviewing each other’s work to ensure compatibility across our implementations.
The embedding vectors we produced allow natural language to be represented in a mathematical form and later used for similarity computation. This functionality is the foundation of RAG and the basis for how the entire system retrieves information.

# How it related to our overall system design
The index file enables the system to quickly retrieve the natural-language text associated with each vector, accelerating information access and preventing the system from getting lost in long, exhaustive searches within a large information repository.