import pandas as pd
import matplotlib.pyplot as plt
def save_feature_previews(df: pd.DataFrame, target: pd.Series, output_dir: str ):
 
    for feature in df.columns:
        plt.figure()
        plt.scatter(target, df[feature], alpha=0.6)
        plt.xlabel(target.name or "Target")
        plt.ylabel(feature)
        plt.title(f"{feature} vs {target.name or 'Target'}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}-{feature}.png".replace(' ', ''))
        plt.close()

    print(f"Saved {len(df.columns)} figures to '{output_dir}'")
