import openai
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

openai.api_key = os.environ['OPENAI_API_KEY']

from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI

# Use local embeddings + gpt-3.5-turbo-16k
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-16k", max_tokens=512, temperature=0.1),
    embed_model="local:BAAI/bge-base-en"
)

set_global_service_context(service_context)

from llama_index import SimpleDirectoryReader, Document
from llama_index.node_parser import HierarchicalNodeParser, SimpleNodeParser, get_leaf_nodes
from llama_index.schema import MetadataMode

from markdown_docs_reader import MarkdownDocsReader

# function that consumes the MarkdownDocsReader to load the markdown docs
def load_markdown_docs(filepath, hierarchical=True):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath, 
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True
    )

    documents = loader.load_data()

    if hierarchical:
        # combine all documents into one
        documents = [
            Document(text="\n\n".join(
                    document.get_content(metadata_mode=MetadataMode.ALL) 
                    for document in documents
                )
            )
        ]

        # chunk into 3 levels
        # majority means 2/3 are retrieved before using the parent
        large_chunk_size = 1536
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                large_chunk_size, 
                large_chunk_size // 3,
            ]
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes, get_leaf_nodes(nodes)
    else:
        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes