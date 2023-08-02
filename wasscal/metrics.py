import torch
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from torchmetrics.functional.classification import binary_calibration_error
from torchmetrics.functional.classification import multiclass_calibration_error


def validate_predictions_for_binning_(pred, y):
    """
    Binning checks
    :param pred:
    :param y:
    :return:
    """
    if pred.ndim != 2:
        raise ValueError("The following argument must have only "
                         "have two dimmensions: \n {pred_arg} \n but "
                         "currently has {ndim:n}".format(
            pred_arg=pred,
            ndim=pred.ndim
        ))

    if y.ndim != 1 and (y.ndim == 2 and y.shape[1] != 1):
        print(pred.ndim != 1)
        print(pred.ndim == 2 and pred.shape[1] != 1)
        raise ValueError("The following argument be formatted of shape (n,) or "
                         "like (n x 1) \n {target_arg}:".format(
            target_arg=y,
        ))

    if pred.shape[0] != y.size:
        raise ValueError("Size mismatch between vector of shape {pred_shape} and"
                         "vector of shape {y_shape}: one has too many/too few rows".format(
            pred_shape=pred.shape,
            y_shape=y.shape
        ))

    if np.any(y.astype(np.int32) != y):
        raise ValueError("Labels can only contain integers, check "
                         "the following arg: {label_arg}".format(
            label_arg=y
        ))

def calibration_error(preds, targets, num_bins=15, norm="l1"):
    validate_predictions_for_binning_(preds, targets)
    num_classes = preds.shape[1]

    #This line is added because of a weird bug in the torchmetrics CE code
    #it starts acting funny when you have one of the endpoints 0 or 1
    if not torch.is_tensor(preds):
        preds = torch.clamp(torch.Tensor(preds.copy()), min=torch.finfo().eps, max=1 - torch.finfo().eps)

    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)


    # if num_pred_classes != num_classes:
    #     warnings.warn("There are {num_pred:n} classes being predicted in this "
    #                   "classification task but only {unique_classes:n} in the sample".format(
    #         num_pred=num_classes,
    #         unique_classes=num_pred_classes))


    if num_classes > 2:
        return multiclass_calibration_error(preds=preds,
                                            target=targets,
                                            num_classes=num_classes,
                                            n_bins=num_bins,
                                            norm=norm).numpy()

    else:
        pos_class_pred = preds[:,1].contiguous()

        return binary_calibration_error(preds=pos_class_pred,
                                        target=targets,
                                        n_bins=num_bins,
                                        norm=norm).numpy()




def _toplabel_ece_help(probs, target, k, num_bins, norm):
    pred = np.argmax(probs,axis=1)
    k_index = np.argwhere(pred == k).reshape(-1)
    frac_k = k_index.size/target.size
    return frac_k*calibration_error(probs[k_index] == k, target[k_index], num_bins=num_bins, norm=norm)


def toplabel_ece(probs, true_class, num_bins=15, norm="l1"):
    labels = np.unique(true_class)
    cw_ece = np.asarray([_toplabel_ece_help(probs=probs,
                                      target=true_class,
                                      k=k,
                                      num_bins=num_bins,
                                      norm=norm) for k in labels])

    return cw_ece.sum()

def classwise_ece(probs, target, num_bins, norm, weighted=False):
    labels = np.unique(target)
    ovr_ece = np.asarray([_one_vs_rest_ece(probs=probs,
                         target=target,
                         k=k,
                         num_bins=num_bins,
                         norm=norm) for k in labels])

    if weighted:
        frac = 1/target.shape[0]
        weight = np.bincount(target, weights=np.repeat(frac,target.shape[0]), minlength=probs.shape[1])
    else:
        weight = 1/probs.shape[1]
    return np.dot(weight, ovr_ece).sum()

def _one_vs_rest_ece(probs,target,num_bins, norm, k):
    k_probs, k_target = one_vs_rest(probs, target, k)
    return calibration_error(k_probs, k_target, num_bins=num_bins, norm=norm)


def one_vs_rest(probs, target, k):
    k_probs = probs[:,k]
    k_vs_rest_prob = np.stack([1 -k_probs, k_probs], axis=1)
    k_vs_rest_target = (target == k).astype(np.int32)

    return k_vs_rest_prob, k_vs_rest_target

def one_hot_encode(target):
    size = target.size
    one_hot = np.zeros(shape=(size,target.max() + 1))
    one_hot[np.arange(size),target] = 1
    return one_hot

def brier_scores(pred, target):
    one_hot = one_hot_encode(target)
    return mean_squared_error(one_hot, pred)

def accuracy(pred, target):
    prediction = np.argmax(pred,axis=1)
    return np.equal(prediction, target).mean()

def nll(pred, target):
    return log_loss(target, y_pred=pred)


