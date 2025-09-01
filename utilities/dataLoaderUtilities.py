"""
This script contains the functions required for loading the 1D data for tuning and training of the deep learning models.
"""

import numpy as np


def load_data_simon(file_path, num_samples_per_class=None, num_features=None):
    """
        This function is used for loading and processing the SIMON32/64 dataset.
        :param file_path: path of the .npz file containing dataset
        :return: features, and labels
        """
    print(f"loading the dataset from {file_path} ...")
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    x = data['cipher_text_binary']  # Feature data
    x = np.array([[int(ch) for ch in s] for s in x])
    y = data['plain_text']  # Labels

    # Print the shape of the features and the first 5 data points
    print("Shape of the features:", x.shape)
    print("First 5 feature data points:\n", x[:5])
    print("First 5 label data points:\n", y[:5])

    # Print the shape of the features and the first 5 data points
    print("Shape of the features:", x.shape)
    print("First 5 feature data points:\n", x[:5])
    print("First 5 label data points:\n", y[:5])

    # Process selection of samples per class
    unique_classes = np.unique(y)  # Unique class labels

    # number of unique classes in the dataset
    num_classes = len(unique_classes)

    final_x = []
    final_y = []

    for cls in unique_classes:
        indices = np.where(y == cls)[0]
        if num_samples_per_class is None:
            # If num_samples_per_class is None, use all samples for that class
            selected_indices = indices
        elif num_samples_per_class >= indices.size:
            # If num_samples_per_class is more than available samples, use all samples
            selected_indices = indices
        else:
            # Otherwise, randomly select the specified number of samples
            selected_indices = np.random.choice(indices, num_samples_per_class, replace=False)

        final_x.append(x[selected_indices])
        final_y.append(y[selected_indices])

    # Concatenate results from all classes
    final_x = np.vstack(final_x)
    final_y = np.concatenate(final_y)

    # Finding the unique elements and their counts
    print("unique classes and counts after processing the dataset ...")
    unique_classes, counts = np.unique(final_y, return_counts=True)

    # Printing the unique classes and their counts
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} occurrences")

    return final_x, final_y, num_classes


class DataLoader1D:
    """
    This class is used for loading the 1D data required for tuning, training, and testing of the deep learning models.

    The data files should be in the ".npz" format where each file contains two keys: "x", and "y". The "x"
    represents the features of 1D data for training tne model, and "y" represents the labels corresponding to the 1D
    data.

    Note:- if number of classes is None, all the samples for each class will be selected.
    """

    def __init__(self, filepath, num_samples_per_class=None, num_features=None):
        """
        Initializes the DataLoader1DBinary object.

        Args:
        filepath (str): Path to the .npz file containing the data.
        num_features (int): Number of features to select from each data point.
        num_samples_per_class (int): Number of samples to select for each class.
        """
        self.filepath = filepath
        self.num_features = num_features
        self.num_samples_per_class = num_samples_per_class

    def load_data_binary(self):
        """
        This function is used for loading the 1D dataset that is in the binary format. Meaning, the dataset for
        which the features are in binary format "0", and "1", or "0", and "-1", etc. This function can be used for
        loading that type of dataset.

        Loads and processes the data from the .npz file.

        Returns:
        numpy.ndarray: The processed data.
        """
        print(f"loading the dataset from {self.filepath} ...")
        # Load data from the .npz file
        data = np.load(self.filepath, allow_pickle=True)
        x = data['x']  # Feature data
        y = data['y']  # Labels

        # Determine which features to select
        if self.num_features is None or self.num_features >= x.shape[1]:
            selected_features = range(x.shape[1])  # Use all features
            if self.num_features is not None:
                print(
                    f"Requested number of features is {self.num_features}, but only {x.shape[1]} available. "
                    f"Selecting all features.")
        else:
            selected_features = range(self.num_features)  # Select first num_features features

        x = x[:, selected_features]

        # Print the shape of the features and the first 5 data points
        print("Shape of the features:", x.shape)
        print("First 5 feature data points:\n", x[:5])
        print("First 5 label data points:\n", y[:5])

        # Process selection of samples per class
        unique_classes = np.unique(y)  # Unique class labels

        # number of unique classes in the dataset
        num_classes = len(unique_classes)

        final_x = []
        final_y = []

        for cls in unique_classes:
            indices = np.where(y == cls)[0]
            if self.num_samples_per_class is None:
                # If num_samples_per_class is None, use all samples for that class
                selected_indices = indices
            elif self.num_samples_per_class >= indices.size:
                # If num_samples_per_class is more than available samples, use all samples
                selected_indices = indices
            else:
                # Otherwise, randomly select the specified number of samples
                selected_indices = np.random.choice(indices, self.num_samples_per_class, replace=False)

            final_x.append(x[selected_indices])
            final_y.append(y[selected_indices])

        # Working code with balanced classes ONLY
        # for cls in unique_classes:
        #     indices = np.where(y == cls)[0]
        #     print(f"Number of samples in class {cls}: {indices.size}")  # Print the number of samples in each class
        #     if self.num_samples_per_class is None or self.num_samples_per_class >= indices.size:
        #         final_x.append(x[indices])
        #         final_y.append(y[indices])
        #     else:
        #         selected_indices = np.random.choice(indices, self.num_samples_per_class, replace=False)
        #         final_x.append(x[selected_indices])
        #         final_y.append(y[selected_indices])

        # Concatenate results from all classes
        final_x = np.vstack(final_x)
        final_y = np.concatenate(final_y)

        # Finding the unique elements and their counts
        print("unique classes and counts after processing the dataset ...")
        unique_classes, counts = np.unique(final_y, return_counts=True)

        # Printing the unique classes and their counts
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} occurrences")

        return final_x, final_y, num_classes


    def load_data_binary_key_recovery(self):
        """
        This function is used for loading the 1D dataset that is in the binary format. Meaning, the dataset for
        which the features are in binary format "0", and "1", or "0", and "-1", etc. This function can be used for
        loading that type of dataset.

        Loads and processes the data from the .npz file.

        Returns:
        numpy.ndarray: The processed data.
        """
        print(f"loading the dataset from {self.filepath} ...")
        # Load data from the .npz file
        data = np.load(self.filepath, allow_pickle=True)
        x = data['x']  # Feature data
        y = data['y']  # Labels
        round_key_1 = data['round_keys'][0]
        round_key_3 = data['round_keys'][1]

        # Determine which features to select
        if self.num_features is None or self.num_features >= x.shape[1]:
            selected_features = range(x.shape[1])  # Use all features
            if self.num_features is not None:
                print(
                    f"Requested number of features is {self.num_features}, but only {x.shape[1]} available. "
                    f"Selecting all features.")
        else:
            selected_features = range(self.num_features)  # Select first num_features features

        x = x[:, selected_features]

        # Print the shape of the features and the first 5 data points
        print("Shape of the features:", x.shape)
        print("First 5 feature data points:\n", x[:5])
        print("First 5 label data points:\n", y[:5])

        # Process selection of samples per class
        unique_classes = np.unique(y)  # Unique class labels

        # number of unique classes in the dataset
        num_classes = len(unique_classes)

        final_x = []
        final_y = []

        for cls in unique_classes:
            indices = np.where(y == cls)[0]
            if self.num_samples_per_class is None:
                # If num_samples_per_class is None, use all samples for that class
                selected_indices = indices
            elif self.num_samples_per_class >= indices.size:
                # If num_samples_per_class is more than available samples, use all samples
                selected_indices = indices
            else:
                # Otherwise, randomly select the specified number of samples
                selected_indices = np.random.choice(indices, self.num_samples_per_class, replace=False)

            final_x.append(x[selected_indices])
            final_y.append(y[selected_indices])

        # Working code with balanced classes ONLY
        # for cls in unique_classes:
        #     indices = np.where(y == cls)[0]
        #     print(f"Number of samples in class {cls}: {indices.size}")  # Print the number of samples in each class
        #     if self.num_samples_per_class is None or self.num_samples_per_class >= indices.size:
        #         final_x.append(x[indices])
        #         final_y.append(y[indices])
        #     else:
        #         selected_indices = np.random.choice(indices, self.num_samples_per_class, replace=False)
        #         final_x.append(x[selected_indices])
        #         final_y.append(y[selected_indices])

        # Concatenate results from all classes
        final_x = np.vstack(final_x)
        final_y = np.concatenate(final_y)

        # Finding the unique elements and their counts
        print("unique classes and counts after processing the dataset ...")
        unique_classes, counts = np.unique(final_y, return_counts=True)

        # Printing the unique classes and their counts
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} occurrences")

        return final_x, final_y, num_classes, round_key_1, round_key_3
