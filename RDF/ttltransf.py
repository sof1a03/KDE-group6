from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, BNode
import csv
import os
import time
import numpy as np
import pandas as pd

# Define namespaces
EX = Namespace("http://example.org/owlshelves#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

# Create RDF graph
g = Graph()
g.bind("ex", EX)
g.bind("dc", DC)
g.bind("foaf", FOAF)
g.bind("owl", OWL)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)

# Define ontology metadata
g.add((EX.ontology, RDF.type, OWL.Ontology))
g.add((EX.ontology, RDFS.comment, Literal("Ontology for representing books, users, and ratings in a library system.")))
g.add((EX.ontology, DC.creator, Literal("Group6")))
g.add((EX.ontology, DC.title, Literal("OWL Book Ontology")))
g.add((EX.ontology, DC.date, Literal("2025-01-20", datatype=XSD.date)))
g.add((EX.ontology, RDFS.seeAlso, Literal("https://github.com/ayesha-banu79/Owl-Ontology/blob/master/Library%20Ontology.owl")))

# Define classes with detailed annotations
classes = {
    "Book": "Represents a book entity in the library system.",
    "User": "Represents a user who interacts with the library system.",
    "Rating": "Represents a user's rating for a book.",
}
for cls, comment in classes.items():
    g.add((EX[cls], RDF.type, OWL.Class))
    g.add((EX[cls], RDFS.comment, Literal(comment)))
    g.add((EX[cls], RDFS.label, Literal(cls)))

# Define datatype properties with detailed annotations
datatype_properties = {
    "hasISBN": ("Book", XSD.string, "Unique identifier for a book (ISBN)."),
    "hasTitle": ("Book", XSD.string, "Title of the book."),
    "hasAuthor": ("Book", XSD.string, "Author of the book."),
    "hasPublisher": ("Book", XSD.string, "Publisher of the book."),
    "hasYearOfPublication": ("Book", XSD.float, "Year the book was published."),
    "hasBookCover": ("Book", XSD.string, "URL for the book's cover image."),
    "hasGenre": ("Book", XSD.string, "Genre or category of the book."),
    "hasUserID": ("User", XSD.string, "Unique identifier for a user."),
    "hasAge": ("User", XSD.float, "Age of the user."),
    "hasCountry": ("User", XSD.string, "Country of the user."),
    "ratingscore": ("Rating", XSD.float, "Score given to a book (1-10)."),
}
for prop, (domain, range_, comment) in datatype_properties.items():
    g.add((EX[prop], RDF.type, OWL.DatatypeProperty))
    g.add((EX[prop], RDFS.domain, EX[domain]))
    g.add((EX[prop], RDFS.range, range_))
    g.add((EX[prop], RDFS.comment, Literal(comment)))
    g.add((EX[prop], RDFS.label, Literal(prop)))

# Declare "hasISBN" as a functional property
g.add((EX.hasISBN, RDF.type, OWL.FunctionalProperty))

# Define object properties with detailed annotations
object_properties = {
    "ratingUser": ("Rating", "User", "Links a rating to the user who gave it."),
    "ratedBook": ("Rating", "Book", "Links a rating to the book being rated."),
}
for prop, (domain, range_, comment) in object_properties.items():
    g.add((EX[prop], RDF.type, OWL.ObjectProperty))
    g.add((EX[prop], RDFS.domain, EX[domain]))
    g.add((EX[prop], RDFS.range, EX[range_]))
    g.add((EX[prop], RDFS.comment, Literal(comment)))
    g.add((EX[prop], RDFS.label, Literal(prop)))

# Add subclass relationships
g.add((EX.FictionBook, RDF.type, OWL.Class))
g.add((EX.FictionBook, RDFS.subClassOf, EX.Book))
g.add((EX.NonFictionBook, RDF.type, OWL.Class))
g.add((EX.NonFictionBook, RDFS.subClassOf, EX.Book))

# Add restrictions for ISBN
isbn_restriction = BNode()
g.add((isbn_restriction, RDF.type, OWL.Restriction))
g.add((isbn_restriction, OWL.onProperty, EX.hasISBN))
g.add((isbn_restriction, OWL.cardinality, Literal(1, datatype=XSD.integer)))
g.add((EX.Book, RDFS.subClassOf, isbn_restriction))

# Serialize the ontology
print("Starting processing books")
start_time = time.time()
def process_books(csv_file):
    with open(csv_file, 'r',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            book_uri = EX[row["ISBN"]]
            g.add((book_uri, RDF.type, EX.Book))
            g.add((book_uri, EX.hasISBN, Literal(row["ISBN"], datatype=XSD.string)))
            g.add((book_uri, EX.hasTitle, Literal(row["Book_Title"], datatype=XSD.string)))
            g.add((book_uri, EX.hasAuthor, Literal(row["Book_Author"], datatype=XSD.string)))
            if row.get("Year_Of_Publication") and row["Year_Of_Publication"].isdigit():
              g.add((book_uri, EX.hasYearOfPublication, Literal(row["Year_Of_Publication"], datatype=XSD.float)))
            g.add((book_uri, EX.hasPublisher, Literal(row["Publisher"], datatype=XSD.string)))
            if row.get("Genre").strip():
              g.add((book_uri, EX.hasGenre, Literal(row.get("Genre", "Unknown"), datatype=XSD.string)))
            if row.get("Book_Cover").strip():
              g.add((book_uri, EX.hasBookCover, Literal(row.get("Book_Cover", ""), datatype=XSD.string)))
process_books("books.csv")
end_time = time.time()
elapsed_time = end_time - start_time
print("Books have been processed")
print(f"in {elapsed_time:.2f} seconds")

print("Starting processing users")
start_time = time.time()
def process_users(csv_file):
    with open(csv_file, 'r',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_uri = EX[row["User-ID"]]
            g.add((user_uri, RDF.type, EX.User))
            g.add((user_uri, EX.hasUserID, Literal(row["User-ID"], datatype=XSD.string)))
            if row.get("Age") and row["Age"].isdigit():
              g.add((user_uri, EX.hasAge, Literal(row["Age"], datatype=XSD.float)))
            g.add((user_uri, EX.hasAgeGroup, Literal(row["Age_group"], datatype=XSD.string)))
            g.add((user_uri, EX.hasCountry, Literal(row["Country"], datatype=XSD.string)))
process_users("users.csv")
end_time = time.time()
elapsed_time = end_time - start_time
print("Users have been processed")
print(f"in {elapsed_time:.2f} seconds")

print("Starting processing ratings")
start_time = time.time()
def process_ratings(csv_file):
    with open(csv_file, 'r',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rating_uri = EX[f"rating_{row['User-ID']}_{row['ISBN']}"]
            g.add((rating_uri, RDF.type, EX.Rating))
            g.add((rating_uri, EX.ratingUser, EX[row["User-ID"]]))
            g.add((rating_uri, EX.ratedBook, EX[row["ISBN"]]))
            g.add((rating_uri, EX.ratingscore, Literal(row["Book-Rating"], datatype=XSD.float)))
process_ratings("ratings.csv")

end_time = time.time()
elapsed_time = end_time - start_time
print("Ratings have been processed")
print(f"in {elapsed_time:.2f} seconds")

# Save file
print("Starting serialization")
start_time = time.time()
g.serialize(destination='owlshelves.ttl', format='turtle')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Serialization completed in {elapsed_time:.2f} seconds")
print("RDF data has been written to ttl")

# Compute and display dataset statistics
def dataset_statistics(graph):

    num_triples = len(graph)# Number of triples
    resources = set(graph.subjects()).union(graph.objects())
    num_resources = len(resources) # Number of unique resources (subjects and objects)

    # Number of unique properties
    properties = set(graph.predicates())
    num_properties = len(properties)

    # Print statistics
    print(f"Dataset Statistics:")
    print(f" - Number of triples: {num_triples}")
    print(f" - Number of unique resources: {num_resources}")
    print(f" - Number of unique properties: {num_properties}")

dataset_statistics(g)

def compute_size_difference(books_file, users_file, ratings_file, ontology_file):
    """
    Compute and display the size differences between the datasets and the ontology file.

    Parameters:
        books_file (str): Path to the books CSV file.
        users_file (str): Path to the users CSV file.
        ratings_file (str): Path to the ratings CSV file.
        ontology_file (str): Path to the serialized ontology file in TTL format.
    """
    # Get file sizes
    books_size = os.path.getsize(books_file)
    users_size = os.path.getsize(users_file)
    ratings_size = os.path.getsize(ratings_file)
    ontology_size = os.path.getsize(ontology_file)

    # Compute differences
    total_dataset_size = books_size + users_size + ratings_size
    size_difference = ((total_dataset_size - ontology_size) / total_dataset_size) * 100

    # Print results
    print("File Sizes (in bytes):")
    print(f" - Books dataset: {books_size}")
    print(f" - Users dataset: {users_size}")
    print(f" - Ratings dataset: {ratings_size}")
    print(f" - Total dataset size: {total_dataset_size}")
    print(f" - Ontology (TTL) file: {ontology_size}")
    print(f"Size Difference (Total dataset - Ontology): {round(size_difference,2)}%")

compute_size_difference("books.csv", "users.csv", "ratings.csv", "owlshelves.ttl")