import argparse
from data_loader import load_trials, build_full_text
from tfidf_model import TFIDFModel
from hybrid_model import HybridModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "query"], default="query")
    parser.add_argument("--text", type=str, default="kidney disease anemia ferric citrate")
    args = parser.parse_args()

    df = load_trials("data/compact_trials.csv")
    df = build_full_text(df)

    tfidf = TFIDFModel().train(df)
    hybrid = HybridModel(tfidf, alpha=0.7, beta=0.3)

    if args.mode == "query":
        results = hybrid.query(args.text, top_k=5)
        for r in results:
            print(r)

    elif args.mode == "train":
        tfidf.save()
        print("TF-IDF model saved.")

if __name__ == "__main__":
    main()
