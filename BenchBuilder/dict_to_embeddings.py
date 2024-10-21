import argparse
import pickle
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/input.pkl")
    parser.add_argument("--output_file", type=str, default="data/output.npy")
    args = parser.parse_args()

    with open(args.input_file, "rb") as f:
        embeddings_data = pickle.load(f)

    text_embeddings = embeddings_data['text_embeddings']
    image_embeddings = embeddings_data['image_embeddings']

    # Ensure the embeddings have the same length
    assert len(text_embeddings) == len(image_embeddings), "Text and image embeddings must have the same length"

    # Calculate the average of text and image embeddings
    averaged_embeddings = np.mean([text_embeddings, image_embeddings], axis=0)

    print(averaged_embeddings.shape)

    # Save the averaged embeddings as a numpy array
    np.save(args.output_file, averaged_embeddings)

    print(f"Averaged embeddings saved to {args.output_file}")
