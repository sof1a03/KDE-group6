from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from pykeen.triples import TriplesFactory
from node2vec import Node2Vec
import torch
import random

app = FastAPI()

# Configuration for GraphDB and file paths
GRAPHDB_SPARQL_ENDPOINT = "http://localhost:3030/owlshelvesbig/sparql" #needs to put the db endpoint
TRANSE_MODEL_DIR = "transe_model_output" # Directory where the TransE model is stored
NODE2VEC_EMBEDDINGS_PATH = "node2vec_embeddings.vec" # File path to Node2Vec embeddings
DEFAULT_PAGE_SIZE = 10  # Default number of results per page for pagination

# Initialize TransE and Node2Vec models
transe_model = None
node2vec_model = None

# Load TransE model
def load_transe_model():
    """
    Load the pre-trained TransE model from the specified directory.
    """
    global transe_model
    model_path = f"{TRANSE_MODEL_DIR}/trained_model.pkl"# Path to the serialized TransE model
    transe_model = torch.load(model_path, map_location=torch.device("cpu")) # Load the model to CPU
    
    
# Load Node2Vec embeddings
def load_node2vec_embeddings():
    """
    Load precomputed Node2Vec embeddings as a Word2Vec model.
    """
    global node2vec_model
    # Assume embeddings are precomputed and available as a Word2Vec model
    from gensim.models import KeyedVectors # Import KeyedVectors for Word2Vec model
    node2vec_model = KeyedVectors.load_word2vec_format(NODE2VEC_EMBEDDINGS_PATH, binary=False) # Load embeddings
    
    
# Helper to execute SPARQL queries
def execute_sparql_query(query: str):
    """
    Execute a SPARQL query against the GraphDB endpoint.
    Args:
        query (str): The SPARQL query string.
    Returns:
        dict: Parsed JSON response from GraphDB.
    """
    sparql = SPARQLWrapper(GRAPHDB_SPARQL_ENDPOINT)# Initialize SPARQL wrapper with the endpoint URL
    sparql.setQuery(query) # Set the SPARQL query
    sparql.setReturnFormat(JSON)# Execute the query and parse the JSON response
    try:
        response = sparql.query().convert() # Execute the query and parse the JSON response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL query failed: {e}")  # Handle errors


# TransE-based recommendation logic
def predict_top_books_transe(user_id, top_n=5):
    """
    Predict the top book recommendations for a user based on TransE embeddings.
    Args:
        user_id (str): The ID of the user.
        top_n (int): Number of recommendations to return.
    Returns:
        list: A list of recommended book IDs.
    """
    if not transe_model:  # Ensure the TransE model is loaded
        raise HTTPException(status_code=500, detail="TransE model not loaded")

    entity_to_id = transe_model.entity_to_id # Retrieve entity-to-ID mapping from the model
    if user_id not in entity_to_id:  # Check if the user exists in the model
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get the embedding for the user
    user_embedding = transe_model.entity_representations[0](
        torch.tensor([entity_to_id[user_id]])
    ).detach().numpy().squeeze()

    ''' ATTENTION: I'M NOT SURE THIS PART IS CORRECT, NEED TO CHECK WITH THE TEAM'''
    similarities = [] # List to store similarity scores
    for entity, idx in entity_to_id.items():
        if entity.startswith("http://example.org/book"):  # Filter only books
            entity_embedding = transe_model.entity_representations[0](
                torch.tensor([idx])
            ).detach().numpy().squeeze()
            similarity = -((user_embedding - entity_embedding) ** 2).sum()
            similarities.append((entity, similarity))

    # Sort by similarity and return the top N books
    top_books = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return [book for book, _ in top_books]


# Node2Vec-based similar book recommendations
def predict_similar_books_node2vec(book_id, top_n=5):
    """
    Recommend books similar to a given book using Node2Vec embeddings.
    Args:
        book_id (str): The ID of the book.
        top_n (int): Number of similar books to return.
    Returns:
        list: A list of similar book IDs.
    """
    if not node2vec_model: # Ensure the Node2Vec model is loaded
        raise HTTPException(status_code=500, detail="Node2Vec model not loaded")

    try:
        similar_books = node2vec_model.most_similar(book_id, topn=top_n)
        return [book for book, _ in similar_books] # Return only the book IDs
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found in Node2Vec embeddings")

# API endpoint: Fetch personalized recommendations
@app.get("/api/recommended_books")
def recommended_books(userid: str, top_n: int = 5):
    """
    Fetch personalized book recommendations for a user using TransE and demographics.
    Args:
        userid (str): The ID of the user.
        top_n (int): Number of recommendations to return.
    Returns:
        dict: A dictionary with the user ID and recommended books.
    """
    try:
        # Get recommendations using TransE
        recommendations = predict_top_books_transe(userid, top_n=top_n)
        return {"userid": userid, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# API endpoint: Fetch books similar to a given book
@app.get("/api/similar_books")
def similar_books(bookid: str, top_n: int = 5):
    """
    Fetch books similar to the given book using Node2Vec embeddings.
    Args:
        bookid (str): The ID of the book.
        top_n (int): Number of similar books to return.
    Returns:
        dict: A dictionary with the book ID and similar books.
    """
    try:
        recommendations = predict_similar_books_node2vec(bookid, top_n=top_n) # Get similar books
        return {"bookid": bookid, "similar_books": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint: Search for books
@app.get("/api/search")
def search_books(
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pageSize: int = 10,
    pageNum: int = 1,
):
    query = """
    PREFIX ex: <http://example.org/owlshelves#>
    SELECT ?book ?title ?author ?ISBN ?publisher ?year WHERE {
        ?book ex:hasTitle ?title .
        OPTIONAL { ?book ex:hasAuthor ?author . }
        OPTIONAL { ?book ex:hasISBN ?ISBN . }
        OPTIONAL { ?book ex:hasPublisher ?publisher . }
        OPTIONAL { ?book ex:hasYearOfPublication ?year . }
    """

    if isbn:
        query += f'?book ex:hasISBN "{isbn}"^^xsd:string .'

    if title:
        query += f'FILTER regex(?title, "{title}", "i") .'

    if author:
        query += f'FILTER regex(?author, "{author}", "i") .'

    if start_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year >= {start_year}) .'

    if end_year:
        query += f'?book ex:hasYearOfPublication ?year . FILTER(?year <= {end_year}) .'

    query += f"}} LIMIT {pageSize} OFFSET {(pageNum - 1) * pageSize}"
    print(query)

    try:
        results = execute_sparql_query(query)
        books = []
        for binding in results["results"]["bindings"]:
            book = {
                "bookid": binding["book"]["value"],
                "title": binding["title"]["value"],
                "author": binding["author"]["value"] if "author" in binding else None,
                "ISBN": binding["ISBN"]["value"] if "ISBN" in binding else None,
                "publisher": binding["publisher"]["value"] if "publisher" in binding else None,
                "year": binding["year"]["value"] if "year" in binding else None
            }
            books.append(book)
        return {"results": books}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# API endpoint: Fetch random book recommendations
@app.get("/api/surprise_me")
def surprise_me(userid: Optional[str] = None, top_n: int = 5):
    """
    Fetch random book recommendations, excluding books already liked by the user.
    Args:
        userid (str, optional): The ID of the user.
        top_n (int): Number of random books to return.
    Returns:
        dict: A dictionary of random book recommendations.
    """
    try:
        query = "SELECT ?book WHERE { ?book a <http://example.org/Book> }"
        all_books = execute_sparql_query(query)["results"]["bindings"]

        if userid: # If a user ID is provided, exclude liked books
            liked_query = f"""
            SELECT ?book WHERE {{
                <http://example.org/user/{userid}> <http://example.org/likesBook> ?book .
            }}
            """
            liked_books = execute_sparql_query(liked_query)["results"]["bindings"]
            liked_book_ids = {b["book"]["value"] for b in liked_books}
            available_books = [b["book"]["value"] for b in all_books if b["book"]["value"] not in liked_book_ids]
        else:
            available_books = [b["book"]["value"] for b in all_books]

        random.shuffle(available_books) # Shuffle books for randomness
        return {"surprise_me": available_books[:top_n]} # Return random recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load models on startup
@app.on_event("startup")
def load_models():
    """
    Load models (TransE and Node2Vec) when the application starts.
    """
    load_transe_model()  # Load the TransE model
    load_node2vec_embeddings()  # Load the Node2Vec embeddings
