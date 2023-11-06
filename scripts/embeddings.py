import pickle

from pokemb.mod import PokEmb
from pokemb.autoencode import encode


def main():
    output_obj = {}

    for size in [32, 64, 128, 256]:
        for gen in range(1, 10):
            output_obj[f"gen{gen}"] = {}
            emb = PokEmb(gen)

            for space in ["species", "abilities", "moves", "items"]:
                embeddings = getattr(emb, space).weight.numpy()

                print(f"gen{gen}", space)
                embeddings = encode(embeddings, embeddings.shape[-1], size, thresh=3e-3)

                output_obj[f"gen{gen}"][space] = embeddings

            print()

        with open(f"pokemb/data/encodings-{size}.pkl", "wb") as f:
            pickle.dump(output_obj, f)


if __name__ == "__main__":
    main()
