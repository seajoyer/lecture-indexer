# Video Lecture Content Indexer

> [!NOTE] 
> 
> *This project was developed with extensive use of AI assistance. The AI wrote almost all of the code based on specifications and requirements, implementing the complex algorithms and system architecture. Use with caution :)*
> 

This system processes YouTube video lectures across various academic disciplines, creating a unified searchable index of concepts. It allows students to quickly find relevant educational content within a large video database by distinguishing between passing mentions and substantive explanations of concepts.

## Technical Innovation

The primary technical innovation is the application of Linear Multiple Longest Common Subsequence (MLCS) algorithms combined with natural language processing methods to identify, analyze, and index academic concepts in video transcripts. The system implements concept signature extraction to determine the core patterns that define educational concepts across multiple videos.

## Components

- **YouTube Data Extraction**: Extract video metadata and transcripts with multilingual support
- **Transcript Processing**: Process raw transcripts with NLP to identify educational segments
- **Concept Extraction**: Match transcript segments against a concept repository
- **Concept Deduplication**: Identify and merge similar concepts using MLCS algorithms
- **Concept Repository**: Store and manage relationships between concepts
- **Search Engine**: Find relevant educational content with advanced ranking
- **Learning Path Generation**: Create optimized paths through concept prerequisites
- **API Service**: Provide access to the indexing and search functionality

## ML/DS Libraries Used

- **Natural Language Processing (NLP)**:
  - NLTK (Natural Language Toolkit)
  - spaCy
  - langdetect

- **Machine Learning**:
  - scikit-learn (sklearn)
  - TF-IDF vectorization
  - Cosine similarity measurements

- **Data Processing**:
  - NumPy (implied in ML components)
  - SQLite (for data storage and retrieval)

- **Algorithm Implementation**:
  - Custom MLCS (Multiple Longest Common Subsequence) implementation
  - Graph algorithms for learning path generation.
