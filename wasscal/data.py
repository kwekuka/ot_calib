import os
import copy
import pickle
import numpy as np
from wasscal import transport
from scipy.special import logit
from scipy.special import softmax


def clip_ep(X):
    eps = np.finfo('float32').eps
    return np.clip(X, eps, 1-eps)

def prob_to_logit(probs):
    clipped_probs = clip_ep(probs)
    logits = logit(clipped_probs)
    return logits


def __validate_logits(X):
    is_logit = (X > 0).any()
    if not is_logit:
        X = prob_to_logit(X)
    return X
def __validate_probs(X):
    is_prob = (X.sum(axis=1) == 1).all()
    if not is_prob:
        X = softmax(X, axis=1)

        #For numerical precision
        X = X / X.sum(axis=1).reshape(-1,1)
    return X

def load(file_path, return_logits=False):
    file_io = pickle.load(open(file_path, 'rb'))
    if type(file_io) is list and len(file_io) > 1:
        source, labels = pickle.load(open(file_path, 'rb'))[-1]
    else:
        source, labels = pickle.load(open(file_path, 'rb'))

    if return_logits:
        source = __validate_logits(source)
    else:
        source = __validate_probs(source)
    return source, labels.reshape(-1)


def load_tt_split(file_path, return_logits=False):
    file_io = pickle.load(open(file_path, 'rb'))
    train_x, train_y = file_io[0]
    test_x, test_y = file_io[1]

    if return_logits:
        train_x = __validate_logits(train_x)
        test_x = __validate_logits(test_x)
    else:
        train_x = __validate_probs(train_x)
        test_x = __validate_probs(test_x)

    return train_x, train_y.reshape(-1), test_x, test_y.reshape(-1)






def supervised_calibrate(file_path, folder, **kwargs):
    """

    :param source_prob:
    :param source_y:
    :param folder:
    :param kwargs:
    :return:
    """
    probs, labels = load(file_path, return_logits=False)

    wasscal = transport.WassersteinCalibration()
    wasscal.fit(probs, labels)
    calibrated = wasscal.calibrate(probs, labels)

    if kwargs.get('save'):
        assert 'file_name' in kwargs and type(kwargs['file_name']) is str

        file_name = kwargs.get('file_name')

        path = f"./data/{folder}"
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, f"{file_name}_source"), probs)
        np.save(os.path.join(path, f"{file_name}_labels"), labels)
        np.save(os.path.join(path, f"{file_name}_calibrated"), calibrated)

    return probs, calibrated, labels

class Sampler:
  def __init__(self, samples, batch_size):
    self.samples = samples
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    dim = self.samples.shape[0]
    index = np.random.choice(dim, size=self.batch_size, replace=True)
    return self.samples[index]


class ParallelSampler:
    def __init__(self, samples, batch_size, randomness=None):
        self.samples = samples
        self.batch_size = batch_size

        if randomness is None:
            randomness = np.arange(samples.shape[0])
        self.randomness = iter(randomness)

    def __iter__(self):
        return self

    def __next__(self):
        seed = next(self.randomness)
        rng = np.random.default_rng(seed=seed)
        index = rng.choice(a=np.arange(self.samples.shape[0]), size=self.batch_size, replace=True)
        return self.samples[index]





