import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def display_basic_stats(loader, which: str = 'train', show_percent: bool = True) -> dict:
    _, y = loader.get_Xy(which=which, flat=False, normalize=False)
    
    counts = np.bincount(y, minlength=10)
    total = int(counts.sum())
    
    print(f"--- MNIST {which.upper()} dataset basic stats ---")
    print(f"Total samples: {total}")
    for digit in range(10):
        c = int(counts[digit])
        if show_percent:
            percent = (c / total * 100) if total > 0 else 0.0
            print(f"{digit}: {c:5d}  ({percent:5.1f}%)")
        else:
            print(f"{digit}: {c:5d}")
    print("\n")
    return {"total": total, "counts": counts}
        

def display_digits_distribution_percent_graph(loader, which: str = 'train', show_percent: bool = True) -> None:
    _, y = loader.get_Xy(which=which, flat=False, normalize=False)
    
    if show_percent:
        counts = np.bincount(y, minlength=10)
        total = int(counts.sum())
        percent = (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=np.float32)
        df = pd.DataFrame({
            "digit": np.arange(10),
            "count": counts.astype(int),
            "percent": percent,
        })
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df, x="digit", y="count", color="#4C72B0")
        ax.set_title(f"MNIST {which.upper()} - Répartition des digits")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Nombre d'exemplaires")
        for i, row in df.iterrows():
            ax.text(i, row["count"], f"{row['percent']:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        sns.countplot(x=y)
    plt.show()


def show_image(loader, index: int, which: str = 'train') -> int:
    X, y = loader.get_Xy(which=which, flat=False, normalize=False)

    if index < 0 or index >= X.shape[0]:
        raise IndexError(f"Index {index} out of bounds for {which.upper()} (0..{X.shape[0]-1})")

    img = X[index]
    label = int(y[index])
    
    plt.imshow(img, cmap="gray")
    plt.title(f"MNIST {which.upper()} - index={index} - label={label}")
    plt.axis("off")
    plt.show()
    
    return label
    

def mean_image_per_digit(loader, which: str = 'train') -> np.ndarray:
    X, y = loader.get_Xy(which=which, flat=False, normalize=False)
    
    means = np.zeros((10, 28, 28), dtype=np.float32)
    
    for digit in range(10):
        digit_imgs = X[y == digit]
        if digit_imgs.shape[0] == 0:
            raise ValueError(f"No example of the digit {digit} in {which.upper()}")
        means[digit] = digit_imgs.mean(axis=0)
    
    return means


def show_mean_digits(loader, which: str = 'train') -> None:
    means = loader.mean_image_per_digit(loader, which)
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    
    for d in range(10):
        axes[d].imshow(means[d], cmap="gray", vmin=0, vmax=255)
        axes[d].set_title(f"{d}")
        axes[d].axis("off")

    fig.suptitle(f"MNIST {which.upper()} - Mean image per digit")
    plt.tight_layout()
    plt.show()
        
