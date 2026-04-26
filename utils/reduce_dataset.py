import pandas as pd

def reduce_liar_dataset(input_path, output_path, num_samples=1000):
    df = pd.read_csv(input_path, sep='\t', header=None)
    df_sample = df.sample(n=num_samples, random_state=42)
    df_sample.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"✅ Saved {num_samples} samples to {output_path}")

# Example usage
if __name__ == "__main__":
    reduce_liar_dataset("data/train.tsv", "data/train_small.tsv", num_samples=1000)
    reduce_liar_dataset("data/valid.tsv", "data/valid_small.tsv", num_samples=200)
