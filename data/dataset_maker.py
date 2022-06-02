# Adapted from https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f

from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
SPLITS = [ [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9]]
KNOWN_SPLITS = [[0,1,2,4,5,9],
        [0,3,5,7,8,9],
        [0,1,5,6,7,8],
        [3,4,5,7,8,9],
        [0,1,2,3,7,8]]
KNOWN_SPLIT_NAMES = [[CIFAR_CLASSES[i] for i in s] for s in KNOWN_SPLITS]



# Define a function to separate CIFAR classes by class index
def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transforms):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transforms = transforms

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transforms(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


def split_dataset(dataset, split_ind, transforms):
    
    # Separating trainset/testset data/label
    x_set = dataset.data
    y_set = dataset.targets

    split_set = DatasetMaker(
        [get_class_i(x_set, y_set, classDict[class_name]) for class_name in KNOWN_SPLIT_NAMES[split_ind]], transforms)

    return split_set