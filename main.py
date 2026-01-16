from mnist_loader import MnistDataLoader

train_imgs = "input/train-images.idx3-ubyte"
train_labels = "input/train-labels.idx1-ubyte"
test_imgs = "input/t10k-images.idx3-ubyte"
test_labels = "input/t10k-labels.idx1-ubyte"

loader = MnistDataLoader(train_imgs, train_labels, test_imgs, test_labels)
loader.load_data()

loader.display_basics_stats("train")
loader.display_basics_stats("test")
# loader.show_image(0, "train")
# loader.show_image(42, "test")
loader.show_mean_digits("train")