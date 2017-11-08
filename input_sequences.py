import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, square

class SequenceDatabase(object):

    def __init__(self, **kwargs):
        self.type = kwargs["dataset_type"]
        self.batch_size = kwargs["batch_size"]
        self.n_points = kwargs["n_points"]
        self.set_prop = kwargs["set_prop"]
        self.sequence_length = kwargs["sequence_length"]
        self.n_examples = kwargs["n_examples"]
        if self.type == "regression":
            self.build_regression()
        elif self.type == "classification":
            self.build_classification()

    def build_regression(self):
        self.x = np.linspace(0, 1, self.n_points)
        # here is where you make the regression function
        self.y = np.sin(16*(2*np.pi)*self.x) + np.cos(26*np.pi*(np.abs(-0.5+self.x))*self.x)
        self.y += np.random.normal(loc=0, scale=0.01, size=self.y.shape)
        #plt.plot(self.x, self.y)
        #plt.show()

        data_indexes = np.arange(self.n_points)
        np.random.shuffle(data_indexes)
        train_indexes = np.sort(data_indexes[:int(self.set_prop[0] * self.n_points)])
        test_indexes = np.sort(data_indexes[int(self.set_prop[0] * self.n_points):])
        self.sets = {}
        self.sets["train"] = {"x": self.x[train_indexes, np.newaxis],
                              "y": self.y[train_indexes]}
        self.sets["test"] = {"x": self.x[test_indexes, np.newaxis],
                             "y": self.y[test_indexes]}

    def build_classification(self):

        def sample_parameters():
            amplitude = np.random.uniform()
            freq = np.random.uniform(low=2, high=8)
            freq2 = np.random.uniform(low=2, high=8)
            phase = np.random.uniform()
            return amplitude, freq, freq2, phase

        self.n_epochs = 0

        x_param = np.linspace(0, 1, self.n_points)
        self.sets = {}
        x = []
        y = []
        signal_types = ["sine", "mod_sine", "sawtooth", "squared"]
        for i in range(int(self.n_examples/4)):
            x_sample = np.random.choice(x_param, size=self.sequence_length, replace=False)
            x_sample = np.sort(x_sample)
            amp, f1, f2, phase = sample_parameters()
            x.append(amp*np.sin(x_sample*(2*np.pi*f1)+phase)[np.newaxis, ...])  # sine wave
            x.append((amp*np.cos(x_sample*(2*np.pi*f2)+phase)*np.sin(x_sample*(2*np.pi*f1)))[np.newaxis, ...])  # modulated sine
            x.append(amp*sawtooth(t=x_sample*(2*np.pi*f1+phase))[np.newaxis, ...])  # sawtooth signal
            x.append(amp*square(t=x_sample*(2*np.pi*f1+phase))[np.newaxis, ...])  # squared signal
            y += [0, ] + [1, ] + [2, ] + [3, ]
        x = np.concatenate(x, axis=0)
        y = np.array(y)
        y = self.labels_to_one_hot(y)

        print(x.shape, y.shape)

        # Sorting
        sort_index = np.arange(start=0, stop=x.shape[0], step=1)
        np.random.shuffle(sort_index)
        x = x[sort_index, ...]
        y = y[sort_index, ...]

        n_train = int(self.n_examples*self.set_prop[0])

        self.sets["train"] = {"x": x[:n_train, ..., np.newaxis],
                              "y": y[:n_train, ...],
                              "n_batch": 0}
        self.sets["test"] = {"x": x[n_train:, ..., np.newaxis],
                             "y": y[n_train:, ...],
                             "n_batch": 0}

    def next_batch(self, batch_size=128, sub_set="train"):
        if self.type == "regression":
            batch_x = []
            batch_y = []
            for i in range(batch_size):
                random_index = np.random.randint(low=0, high=len(self.sets[sub_set]["y"]))
                high_index, low_index = random_index + self.sequence_length/2, random_index - self.sequence_length/2
                if high_index >= len(self.sets[sub_set]["y"]):
                    high_index -= (high_index - len(self.sets[sub_set]["y"]))
                    low_index -= (high_index - len(self.sets[sub_set]["y"]))
                elif low_index < 0:
                    high_index += low_index
                    low_index += low_index
                batch_x.append(self.sets[sub_set]["x"][int(low_index):int(high_index)])
                batch_y.append(self.sets[sub_set]["y"][int(low_index):int(high_index)])
            return np.array(batch_x), np.array(batch_y)

        elif self.type == "classification":
            n_batch = self.sets[sub_set]["n_batch"]
            start_index, end_index = n_batch*batch_size, (n_batch+1)*batch_size
            if end_index >= len(self.sets[sub_set]["y"]):
                if sub_set == "train":
                    self.n_epochs += 1
                    print("n_epochs = "+str(self.n_epochs))
                batch_x = np.concatenate([self.sets[sub_set]["x"][start_index:, ...],
                                          self.sets[sub_set]["x"][:(end_index-len(self.sets[sub_set]["y"])), ...]],
                                         axis=0)
                batch_y = np.concatenate([self.sets[sub_set]["y"][start_index:, ...],
                                          self.sets[sub_set]["y"][:(end_index-len(self.sets[sub_set]["y"])), ...]],
                                         axis=0)
                self.sets[sub_set]["n_batch"] = 0
            else:
                batch_x = self.sets[sub_set]["x"][start_index:end_index, ...]
                batch_y = self.sets[sub_set]["y"][start_index:end_index, ...]
                self.sets[sub_set]["n_batch"] += 1
            return batch_x, batch_y

    def labels_to_one_hot(self, labels):
        """Converts list of integers to numpy 2D array with one-hot encoding"""
        N = len(labels)
        one_hot_labels = np.zeros([N, 4], dtype=int)
        one_hot_labels[np.arange(N), labels.astype(int)] = 1
        return one_hot_labels


if __name__ == "__main__":
    set_prop = [0.8, 0.2]
    n_examples = 500
    database = SequenceDatabase(batch_size=128,
                                dataset_type="classification",
                                set_prop=set_prop,
                                sequence_length=15,
                                n_points=100,
                                n_examples=n_examples)
    for i in range(300):
        batch_x, batch_y = database.next_batch()
        print(i, batch_x.shape, batch_y.shape)

    for i in range(300):
        batch_x, batch_y = database.next_batch(sub_set="test")
        print(i, batch_x.shape, batch_y.shape)