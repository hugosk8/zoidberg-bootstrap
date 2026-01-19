import numpy as np
import struct
import matplotlib.pyplot as plt

class MnistDataLoader(object):
    def __init__( self, train_imgs_path, train_labels_path, test_imgs_path, test_labels_path,):
        self.train_imgs_path = train_imgs_path
        self.train_labels_path = train_labels_path
        self.test_imgs_path = test_imgs_path
        self.test_labels_path = test_labels_path
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_flat = None
        self.X_test_flat = None
        
        
    def read_labels(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            magic, n = struct.unpack('>II', file.read(8))
            if magic != 2049:
                raise ValueError(f"Labels: mismatch magic number {magic} (expected 2049)")
            labels = np.frombuffer(file.read(), dtype=np.uint8)
        if labels.shape[0] != n:
            raise ValueError(f"Labels: mismatch in label count {labels.shape[0]} (expected {n})")
        return labels
    
    
    def read_images(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            magic, n, rows, cols = struct.unpack('>IIII', file.read(16))
            if magic != 2051:
                raise ValueError(f"Images: mismatch magic number {magic} (expected 2051)")
            if rows != 28 or cols != 28:
                raise ValueError(f"Images: unexpected size {rows}x{cols} (expected 28x28)")
            data = np.frombuffer(file.read(), dtype=np.uint8)
            
        expected = n * rows * cols
        if data.shape[0] != expected:
            raise ValueError(f"Images: mismatch in pixel count {data.shape[0]} (expected {expected})")
        return data.reshape(n, rows, cols)
    
    
    def load_data(self) -> None:
        self.X_train = self.read_images(self.train_imgs_path)
        self.y_train = self.read_labels(self.train_labels_path)
        self.X_test = self.read_images(self.test_imgs_path)
        self.y_test = self.read_labels(self.test_labels_path)
        
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("Training data and labels count mismatch")
        if self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError("Test data and labels count mismatch")
        
        
    def flatten_images(self, which: str = 'train', normalize: bool = False) -> np.ndarray:
        if which == 'train':
            X = self.X_train
        elif which == 'test':
            X = self.X_test
        else:
            raise ValueError("which must be 'train' or 'test'")

        if X is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")

        h, w = X.shape[1], X.shape[2]
        X_flat = X.reshape(X.shape[0], h * w)
        if normalize:
            X_flat = X_flat.astype(np.float32) / 255.0
        return X_flat


    def display_basic_stats(self, which: str = 'train', show_percent: bool = True) -> dict:
        if which == 'train':
            y = self.y_train
        elif which == 'test':
            y = self.y_test
        else:
            raise ValueError("which must be 'train' or 'test'")
        
        if y is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")
        
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


    def show_image(self, index: int, which: str = 'train') -> int:
        if which == 'train':
            X, y = self.X_train, self.y_train
        elif which == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("which must be 'train' or 'test'")

        if X is None or y is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")

        if index < 0 or index >= X.shape[0]:
            raise IndexError(f"Index {index} out of bounds for {which.upper()} (0..{X.shape[0]-1})")

        img = X[index]
        label = int(y[index])
        
        plt.imshow(img, cmap="gray")
        plt.title(f"MNIST {which.upper()} - index={index} - label={label}")
        plt.axis("off")
        plt.show()
        
        return label
        

    def mean_image_per_digit(self, which: str = 'train') -> np.ndarray:
        if which == 'train':
            X, y = self.X_train, self.y_train
        elif which == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("which must be 'train' or 'test'")

        if X is None or y is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")
        
        means = np.zeros((10, 28, 28), dtype=np.float32)
        
        for digit in range(10):
            digit_imgs = X[y == digit]
            if digit_imgs.shape[0] == 0:
                raise ValueError(f"No example of the digit {digit} in {which.upper()}")
            means[digit] = digit_imgs.mean(axis=0)
        
        return means
    
    
    def show_mean_digits(self, which: str = 'train') -> None:
        means = self.mean_image_per_digit(which)
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.ravel()
        
        for d in range(10):
            axes[d].imshow(means[d], cmap="gray", vmin=0, vmax=255)
            axes[d].set_title(f"{d}")
            axes[d].axis("off")

        fig.suptitle(f"MNIST {which.upper()} - Mean image per digit")
        plt.tight_layout()
        plt.show()
 
    
    def flatten_images_inplace(self, normalize: bool = False) -> None:
        if self.X_train is None or self.X_test is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")
        
        self.X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        if normalize:
            self.X_train_flat = self.X_train_flat.astype(np.float32) / 255.0
            self.X_test_flat = self.X_test_flat.astype(np.float32) / 255.0
            

