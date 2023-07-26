import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class Bags(Dataset):
    """
    Dataset class creating set of "bags" containing multiple MNIST images
    """
    def __init__(self,
                 label_number: float = 7.0,
                 mean_bag_size: float = 10,
                 std_bag_size: float = 2,
                 num_of_bags: int = 250,
                 seed: int = 0,
                 train: bool = True,
                 ):
        self.label_number = label_number
        self.mean_bag_size = mean_bag_size
        self.std_bag_size = std_bag_size
        self.num_of_bags = num_of_bags
        self.seed = seed
        self.train = train
        self.bags, self.labels = self._generate_bags()

    def _generate_bags(self) -> tuple:
        """
        Generate datapoints (bags) and labels. Bags contain random number of MNIST
        images. Labels (binary classification) are defined in the following way:
            iff bag contains image labelled by self.label_number, label = 1
            else label = 0
        """

        if self.train:
            dataset = datasets.MNIST("./datasets",
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
        else:
            dataset = datasets.MNIST("./datasets",
                                     train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))

        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,)
        mnist_imgs, mnist_labels = next(iter(data_loader))  # nice trick!
        bags, labels = [], []

        # generate lengths of bags
        rng = np.random.RandomState(self.seed)
        lengths = np.round(rng.normal(loc=self.mean_bag_size,
                                      scale=self.std_bag_size,
                                      size=self.num_of_bags)).astype(int)
        # force all lengths to be 5 < len < 250000000
        lengths[lengths < 5] = 5
        lengths[lengths > 250000000] = 250000000
        # In fact, this enforcement together with rounding may change mean and std,
        # possibly a better solution is to use a different distribution (e.g. Beta-
        # -binomial returning integer values).

        for length in lengths:
            idxs = rng.randint(low=0, high=len(dataset), size=length)
            idxs = torch.LongTensor(idxs)
            bag_i = mnist_imgs[idxs]
            label_i = torch.Tensor.int(
                torch.any(mnist_labels[idxs] == self.label_number))

            bags.append(bag_i)
            labels.append(label_i)

        return bags, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int):
        return self.bags[item], self.labels[item]

    def imbalance(self) -> float:
        """
        Compute the sum of labels over the number of labels (=number of bags)
        returning imbalance of the dataset.
        """
        return np.sum(self.labels) / self.num_of_bags


class BalancedBags(Dataset):
    """
    Balanced version of Bags class
    """
    def __init__(self,
                 label_number: float = 7.0,
                 mean_bag_size: float = 10.,
                 std_bag_size: float = 2,
                 num_of_bags: int = 250,
                 seed: int = 0,
                 train: bool = True):
        self.label_number = label_number
        self.mean_bag_size = mean_bag_size
        self.std_bag_size = std_bag_size
        self.num_of_bags = num_of_bags
        self.seed = seed
        self.train = train
        self.bags, self.labels, self.orig_labels = self._generate_bags()

    def _generate_bags(self) -> tuple:

        if self.train:
            dataset = datasets.MNIST("./datasets",
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
        else:
            dataset = datasets.MNIST("./datasets",
                                     train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))

        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, )
        mnist_imgs, mnist_labels = next(iter(data_loader))  # nice trick!

        rng = np.random.RandomState(self.seed)

        bags, labels = [], []
        orig_labels = []
        valid_bags_counter = 0
        label_of_last_bag = 0

        while valid_bags_counter < self.num_of_bags:
            bag_length = np.round(rng.normal(loc=self.mean_bag_size,
                                             scale=self.std_bag_size,
                                             size=1)).astype(int)
            # force length to be 5 < len < 250000000
            if bag_length < 5:
                bag_length = 5
            elif bag_length > 250000000:
                bag_length = 250000000

            indices = torch.LongTensor(rng.randint(low=0, high=len(dataset), size=bag_length))
            labels_in_bag = mnist_labels[indices]

            if torch.any(labels_in_bag == self.label_number) and (label_of_last_bag == 0):
                bag_i = mnist_imgs[indices]
                label_i = torch.Tensor.int(torch.any(labels_in_bag == self.label_number))

                bags.append(bag_i)
                labels.append(label_i)
                orig_labels.append(labels_in_bag)
                label_of_last_bag = 1
                valid_bags_counter += 1

            elif label_of_last_bag == 1:
                index_list = []
                bag_length_counter = 0
                while bag_length_counter < bag_length:
                    index = torch.LongTensor(rng.randint(low=0, high=len(dataset), size=1))
                    label_tmp = mnist_labels[index]
                    if label_tmp != self.label_number:
                        index_list.append(index.item())
                        bag_length_counter += 1

                labels_in_bag = mnist_labels[index_list]
                label_i = torch.Tensor.int(torch.any(labels_in_bag == self.label_number))  # safety for tests..
                bag_i = mnist_imgs[index_list]
                labels.append(label_i)
                bags.append(bag_i)
                orig_labels.append(labels_in_bag)

                label_of_last_bag = 0
                valid_bags_counter += 1

            else:
                pass

        return bags, labels, orig_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int):
        return self.bags[item], self.labels[item]  #, self.orig_labels[item]

    def imbalance(self) -> float:
        """
        Compute the sum of labels over the number of labels (=number of bags)
        returning imbalance of the dataset.
        """
        return np.sum(self.labels) / self.num_of_bags
