from mnist_loader import MnistDataLoader

train_imgs = "input/train-images.idx3-ubyte"
train_labels = "input/train-labels.idx1-ubyte"
test_imgs = "input/t10k-images.idx3-ubyte"
test_labels = "input/t10k-labels.idx1-ubyte"

loader = MnistDataLoader(train_imgs, train_labels, test_imgs, test_labels)
loader.load_data()

# Show digits distribution
# loader.display_basic_stats("train")
# loader.display_basic_stats("test")

# Show selected image
# loader.show_image(0, "train")
# loader.show_image(42, "test")

# # Show mean digits
# loader.show_mean_digits("train")

# # Show flatten images size
# loader.flatten_images_inplace(normalize=True)
# print(loader.X_train_flat.shape)
# print(loader.X_test_flat.shape) 

print(loader.display_digits_distribution_percent_graph())