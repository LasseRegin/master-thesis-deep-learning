
import math
import random

import numpy as np

import data


class BatchLoader:
    def __init__(self, data_loader, batch_size=32):
        self.data_loader = data_loader
        self.batch_size = batch_size

        # Set memmap data file
        self.memmap_file = self.data_loader.data_memmap
        self.data_count = len(self.memmap_file)
        self.batch_count = math.ceil(self.data_count / self.batch_size)

        # Construct indices
        self.indices = []
        for i in range(0, self.batch_count):
            idx_from = (i    ) * self.batch_size
            idx_to   = (i + 1) * self.batch_size
            self.indices.append((idx_from, idx_to))


    def __iter__(self):
        random.shuffle(self.indices)
        for idx_from, idx_to in self.indices:
            # Fetch data rows
            data_rows = self.memmap_file[idx_from:idx_to]

            batch_dict = {}
            for key, indices in self.data_loader.data_indices.items():
                #if key == 'class_vector':
                    #print(indices)
                    #print(data_rows[:, indices[0]:indices[1]])
                if indices[1] - indices[0] == 1:
                    batch_dict[key] = data_rows[:, indices[0]]
                else:
                    batch_dict[key] = data_rows[:, indices[0]:indices[1]]

            yield batch_dict



class MixedBatchLoader:
    def __init__(self, data_loader_1, data_loader_2,  batch_size=32, mix_ratio=0.5):
        assert batch_size % 2 == 0
        self.data_loader_1 = data_loader_1
        self.data_loader_2 = data_loader_2
        self.batch_size = batch_size
        self.mix_ratio = mix_ratio

        # Determine number of observations for each type in each batch
        batch_observations_1 = self.batch_size * self.mix_ratio
        batch_observations_2 = self.batch_size * (1.0 - self.mix_ratio)

        # Determine the maximum number of possible batches
        self.count_1 = len(self.data_loader_1.data_memmap)
        self.count_2 = len(self.data_loader_2.data_memmap)
        batch_count_1 = math.ceil(self.count_1 / batch_observations_1)
        batch_count_2 = math.ceil(self.count_2 / batch_observations_2)
        self.batch_count = min(batch_count_1, batch_count_2)

        self.indices = []
        for i in range(0, self.batch_count):
            idx_from_1 = math.floor((i    ) * batch_observations_1)
            idx_to_1   = math.floor((i + 1) * batch_observations_1)
            idx_from_2 = math.floor((i    ) * batch_observations_2)
            idx_to_2   = math.floor((i + 1) * batch_observations_2)
            self.indices.append(((idx_from_1, idx_to_1), (idx_from_2, idx_to_2)))

    def __iter__(self):
        indices_1 = np.random.permutation(self.count_1)
        indices_2 = np.random.permutation(self.count_2)

        for (idx_from_1, idx_to_1), (idx_from_2, idx_to_2) in self.indices:
            # Determine indices
            batch_idx_1 = indices_1[idx_from_1:idx_to_1]
            batch_idx_2 = indices_2[idx_from_2:idx_to_2]

            # Fetch data rows
            data_rows_1 = self.data_loader_1.data_memmap[batch_idx_1]
            data_rows_2 = self.data_loader_2.data_memmap[batch_idx_2]

            data_rows = np.vstack((data_rows_1, data_rows_2))

            # Shuffle values
            np.random.shuffle(data_rows)

            batch_dict = {}
            for key, indices in self.data_loader_1.data_indices.items():
                if indices[1] - indices[0] == 1:
                    batch_dict[key] = data_rows[:, indices[0]]
                else:
                    batch_dict[key] = data_rows[:, indices[0]:indices[1]]

            yield batch_dict



class ClassSamplingBatchLoader:
    def __init__(self, batch_loader):
        self.batch_loader = batch_loader

        if isinstance(self.batch_loader, MixedBatchLoader):
            self.data_loader_1 = self.batch_loader.data_loader_1
            self.data_loader_2 = self.batch_loader.data_loader_2
        elif isinstance(self.batch_loader, BatchLoader):
            self.data_loader = self.batch_loader.data_loader
        else:
            raise KeyError('Invalid batch_loader argument provided!')

        self.batch_size = self.batch_loader.batch_size
        self.batch_count = self.batch_loader.batch_count

    def __iter__(self):
        for data_dict in self.batch_loader:
            class_counts = data_dict['class_count']
            rand_values = np.random.rand(class_counts.shape[0])
            class_indices = (class_counts * rand_values).astype('int')

            idx = np.arange(0, class_counts.shape[0])
            data_dict['class_idx'] = data_dict['classes'][idx,class_indices]

            yield data_dict
