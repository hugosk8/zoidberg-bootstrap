from mnist_loader import MnistDataLoader
import mnist_viz as viz

train_imgs = "input/train-images.idx3-ubyte"
train_labels = "input/train-labels.idx1-ubyte"
test_imgs = "input/t10k-images.idx3-ubyte"
test_labels = "input/t10k-labels.idx1-ubyte"

loader = MnistDataLoader(train_imgs, train_labels, test_imgs, test_labels)
loader.load_data()

# Stats (prints)
viz.display_basic_stats(loader, "train")
viz.display_basic_stats(loader, "test")

# Afficher une image
viz.show_image(loader, 0, "train")
viz.show_image(loader, 42, "test")

# Moyenne par digit
viz.show_mean_digits(loader, "train")

# Matrices pour la partie Prediction (n, 784)
X_train, y_train = loader.get_Xy("train", flat=True, normalize=True)
X_test, y_test = loader.get_Xy("test", flat=True, normalize=True)

print(X_train.shape)  # (60000, 784)
print(X_test.shape)   # (10000, 784)

# Graphiques distribution
viz.display_digits_distribution_percent_graph(loader, "train", show_percent=False)
viz.display_digits_distribution_percent_graph(loader, "train", show_percent=True)
