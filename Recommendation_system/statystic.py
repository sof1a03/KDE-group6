import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.models import Model
import rdflib
from tqdm import tqdm
import pickle

# Global variable for triples factory
triples_factory = None

# Load TransE model
def load_transe_model():
    """
    Load the pre-trained TransE model from the specified directory.
    """
    global transe_model
    model_path = "trained_model.pkl"  # Path to the serialized TransE model
    print("Attempting to load the TransE model...")
    try:
        transe_model = torch.load(model_path, map_location=torch.device("cpu"))  # Load the model to CPU
        print("Successfully loaded the TransE model!")
        return transe_model
    except Exception as e:
        raise ValueError(f"Failed to load the TransE model: {e}")

# Load TriplesFactory
def load_triples_factory():
    """
    Load the triples factory from a precomputed file if available, otherwise from the RDF file.
    """
    global triples_factory
    pkl_file_path = "triples_factory.pkl"  # Path to the precomputed triples factory file
    rdf_file_path = "owlshelvesf.ttl"

    if triples_factory is None:
        try:
            print("Attempting to load the precomputed triples factory...")
            with open(pkl_file_path, "rb") as f:
                triples_factory = pickle.load(f)
            print("Triples factory loaded successfully from precomputed file!")
        except FileNotFoundError:
            print("Precomputed triples factory not found. Parsing RDF file...")
            rdf_graph = rdflib.Graph()
            rdf_graph.parse(rdf_file_path, format="turtle")

            print(f"Total triples found: {len(rdf_graph)}")

            triples = [(str(subj), str(pred), str(obj)) for subj, pred, obj in rdf_graph]
            triples_factory = TriplesFactory.from_labeled_triples(triples=triples)

            # Optionally save the triples factory for future use
            with open(pkl_file_path, "wb") as f:
                pickle.dump(triples_factory, f)
            print("Triples factory saved for future use!")

    return triples_factory

def calculate_rmse(triples_factory, model):
    """
    Calculate RMSE based on triples and model predictions, handling index mismatches.
    """
    true_ratings = []
    predicted_ratings = []

    for head, relation, tail in triples_factory.mapped_triples:
        try:
            # Ensure indices are within bounds
            if head >= model.entity_representations[0].max_id or tail >= model.entity_representations[0].max_id:
                continue
            if relation >= model.relation_representations[0].max_id:
                continue

            head_emb = model.entity_representations[0](torch.tensor([head], dtype=torch.long))
            tail_emb = model.entity_representations[0](torch.tensor([tail], dtype=torch.long))
            relation_emb = model.relation_representations[0](torch.tensor([relation], dtype=torch.long))

            # Calculate predicted score
            predicted_score = (head_emb + relation_emb - tail_emb).norm(p=2).item()
            true_ratings.append(1)  # Assume binary classification with 1 for valid triples
            predicted_ratings.append(predicted_score)
        except Exception as e:
            print(f"Error for triple ({head}, {relation}, {tail}): {e}")

    if not true_ratings or not predicted_ratings:
        raise ValueError("No valid triples were processed. Check entity/relation indices.")

    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)

    rmse = np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))
    return rmse

def generate_recommendations(triples_factory, model, top_n=10):
    """
    Generate recommendations using the model, handling index mismatches.
    """
    entity_to_id = triples_factory.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    recommendations = {}
    for entity, entity_id in entity_to_id.items():
        if "user" in entity:  # Assuming "user" entities need recommendations
            if entity_id >= model.entity_representations[0].max_id:
                continue

            user_embedding = model.entity_representations[0](torch.tensor([entity_id], dtype=torch.long))

            scores = []
            for other_entity, other_id in entity_to_id.items():
                if "book" in other_entity:  # Assuming "book" entities are recommended
                    if other_id >= model.entity_representations[0].max_id:
                        continue

                    book_embedding = model.entity_representations[0](torch.tensor([other_id], dtype=torch.long))
                    score = -((user_embedding - book_embedding) ** 2).sum().item()
                    scores.append((other_entity, score))

            sorted_recommendations = [book for book, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
            recommendations[entity] = sorted_recommendations[:top_n]

    return recommendations

def calculate_hamming_distance(recommendations, top_l):
    """
    Calculate Hamming Distance based on recommendations.
    """
    distances = []
    user_ids = list(recommendations.keys())

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user_i = set(recommendations[user_ids[i]][:top_l])
            user_j = set(recommendations[user_ids[j]][:top_l])
            hamming_distance = top_l - len(user_i & user_j)
            distances.append(hamming_distance)

    avg_hamming_distance = np.mean(distances)
    return avg_hamming_distance

def calculate_coverage(recommendations, num_books, top_l):
    """
    Calculate coverage of recommendations.
    """
    unique_books = set()
    for recs in recommendations.values():
        unique_books.update(recs[:top_l])

    coverage = len(unique_books) / num_books
    return coverage

# Main Analysis Function
def analyze_model(top_l=10):
    # Load triples factory
    triples_factory = load_triples_factory()

    # Load TransE model
    transe_model = load_transe_model()

    # Validate embedding dimensions
    print("Validating embeddings...")
    print(f"Entity embeddings: {transe_model.entity_representations[0].max_id}, "
          f"Entities in factory: {len(triples_factory.entity_to_id)}")
    print(f"Relation embeddings: {transe_model.relation_representations[0].max_id}, "
          f"Relations in factory: {len(triples_factory.relation_to_id)}")

    # Calculate RMSE
    print("Calculating RMSE...")
    rmse = calculate_rmse(triples_factory, transe_model)
    print(f"RMSE: {rmse:.4f}")

    # Generate Recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(triples_factory, transe_model, top_n=top_l)

    # Calculate Hamming Distance
    print("Calculating Hamming Distance...")
    hamming_distance = calculate_hamming_distance(recommendations, top_l)
    print(f"Hamming Distance: {hamming_distance:.4f}")

    # Calculate Coverage
    print("Calculating Coverage...")
    num_books = sum(1 for entity in triples_factory.entity_to_id if "book" in entity)
    coverage = calculate_coverage(recommendations, num_books, top_l)
    print(f"Coverage: {coverage:.4f}")

    # Return Analysis Results
    return {
        "RMSE": rmse,
        "Hamming Distance": hamming_distance,
        "Coverage": coverage
    }

# Main Test Script
if __name__ == "__main__":
    try:
        analysis_results = analyze_model(top_l=10)
        print("Analysis Results:")
        for metric, value in analysis_results.items():
            print(f"{metric}: {value:.4f}")
    except Exception as error:
        print(f"Error during analysis: {error}")
