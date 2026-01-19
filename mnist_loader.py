import numpy as np
import struct

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
        
        
    def get_Xy(self, which: str = "train", flat: bool = True, normalize: bool = False):
        if which == "train":
            X, y = self.X_train, self.y_train
        elif which == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("which must be 'train' or 'test'")

        if X is None or y is None:
            raise RuntimeError("Dataset not loaded: call load_data() first")

        if flat:
            X = X.reshape(X.shape[0], -1)

        if normalize:
            X = X.astype(np.float32) / 255.0

        return X, y

