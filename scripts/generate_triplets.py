import json
import random
import os

def create_training_triplets(data_dir, output_json):

    doc_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    triplets = []

    for anchor in doc_ids:
        # Positive: Same document (the model aligns different modalities of it)
        positive = anchor 
        
        # Negative: Randomly pick a different document
        negative = random.choice([d for d in doc_ids if d != anchor])
        
        triplets.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        })

    with open(output_json, 'w') as f:
        json.dump(triplets, f, indent=4)
    print(f"Generated {len(triplets)} triplets saved to {output_json}")