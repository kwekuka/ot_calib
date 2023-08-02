import os
from wasscal import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from jax.experimental import jax2tf



class Logger:
    def __init__(self,
                 dataset,
                 method,
                 num_classes,
                 results_path="./results",
        ):
        self.dataset = dataset
        self.results_path = results_path
        self.method = method
        self.num_classes = num_classes

        self.logs = {}
        self.best_conf_ece = np.inf
        self.best_max_ece = np.inf

        path = os.path.join(self.results_path, self.dataset, self.method)
        if not os.path.exists(path):
            os.makedirs(path)

        id = len([entry for entry in os.listdir(path)])
        self.id = id

    def addEntryJax(self,
                    test_probs,
                    test_labels,
                    train_probs,
                    train_labels,
                    iteration,
                    model=None):


        best_ECE = False

        if model is not None:
            test_output = np.asarray(model.transport(test_probs))
            train_output = np.asarray(model.transport(train_probs))
        else:
            test_output = test_probs
            train_output = train_probs

        log = {
            "Test Accuracy": metrics.accuracy(test_output, test_labels),
            "Test Brier Score": metrics.brier_scores(test_output, test_labels),
            "Test NLL": metrics.nll(test_output, test_labels),
            "Train Accuracy": metrics.accuracy(train_output, train_labels),
            "Train Brier Score": metrics.brier_scores(train_output, train_labels),
            "Train NLL": metrics.nll(train_output, train_labels),
        }

        for bins in [15,25]:
            log[f"Test_ECE_{bins}"] = metrics.calibration_error(test_output, test_labels, num_bins=bins, norm="l1")
            log[f"Test_MCE_{bins}"] = metrics.calibration_error(test_output, test_labels, num_bins=bins, norm="max")
            log[f"Test_TCE_{bins}"] = metrics.toplabel_ece(test_output, test_labels, num_bins=bins, norm="l1")
            log[f"Test_CCE_{bins}"] = metrics.classwise_ece(test_output, test_labels,
                                                            num_bins=bins, norm="l1")


            log[f"Train_ECE_{bins}"] = metrics.calibration_error(train_output, train_labels, num_bins=bins, norm="l1")
            log[f"Train_MCE_{bins}"] = metrics.calibration_error(train_output, train_labels, num_bins=bins, norm="max")
            log[f"Train_TCE_{bins}"] = metrics.toplabel_ece(train_output, train_labels, num_bins=bins, norm="l1")
            log[f"Train_CCE_{bins}"] = metrics.classwise_ece(train_output, train_labels,
                                                             num_bins=bins, norm="l1")



        ece15 = log["Test_ECE_15"]

        if ece15 < self.best_conf_ece:
            self.best_conf_ece = ece15
            best_ECE = True
            print(f"Best ECE so far: {ece15}")



        if best_ECE and model is not None:
            path = os.path.join(self.results_path, self.dataset, f"run_{self.id}")
            if not os.path.exists(path):
                os.makedirs(path)


            filename = os.path.join(path, f'model{self.id}_{iteration}')

            f_tf = jax2tf.convert(model.transport)
            my_model = tf.Module()
            my_model.f = tf.function(f_tf, autograph=False,
                                     input_signature=[tf.TensorSpec(self.num_classes, tf.float32)])
            tf.saved_model.save(my_model, filename,
                                options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

        log["Test_Best_ECE"] = best_ECE

        self.logs[iteration] = log

    def addEntry(self,
                    test_probs,
                    test_labels,
                    train_probs,
                    train_labels,
                    method,
    ):



        test_output = test_probs
        train_output = train_probs

        log = {
            "Method": method,
            "Test Accuracy": metrics.accuracy(test_output, test_labels),
            "Test Brier Score": metrics.brier_scores(test_output, test_labels),
            "Test NLL": metrics.nll(test_output, test_labels),
            "Train Accuracy": metrics.accuracy(train_output, train_labels),
            "Train Brier Score": metrics.brier_scores(train_output, train_labels),
            "Train NLL": metrics.nll(train_output, train_labels),
        }

        for bins in [15,25]:
            log[f"Test_ECE_{bins}"] = metrics.calibration_error(test_output, test_labels, num_bins=bins, norm="l1")
            log[f"Test_MCE_{bins}"] = metrics.calibration_error(test_output, test_labels, num_bins=bins, norm="max")
            log[f"Test_TCE_{bins}"] = metrics.toplabel_ece(test_output, test_labels, num_bins=bins, norm="l1")
            log[f"Test_CCE_{bins}"] = metrics.classwise_ece(test_output, test_labels,
                                                            num_bins=bins, norm="l1")


            log[f"Train_ECE_{bins}"] = metrics.calibration_error(train_output, train_labels, num_bins=bins, norm="l1")
            log[f"Train_MCE_{bins}"] = metrics.calibration_error(train_output, train_labels, num_bins=bins, norm="max")
            log[f"Train_TCE_{bins}"] = metrics.toplabel_ece(train_output, train_labels, num_bins=bins, norm="l1")
            log[f"Train_CCE_{bins}"] = metrics.classwise_ece(train_output, train_labels,
                                                             num_bins=bins, norm="l1")

        self.logs[len(self.logs)] = log

    def save(self, baseline=False):
        df = pd.DataFrame.from_dict(self.logs).T

        if not baseline:
            path = os.path.join(self.results_path, self.dataset, self.method, f"run_{self.id}")
            if not os.path.exists(path):
                os.makedirs(path)
            fname = f"{self.method}_{self.dataset}_{self.id}.csv"
        else:
            path = os.path.join(self.results_path, self.dataset, self.method, "baseline")
            if not os.path.exists(path):
                os.makedirs(path)
            baselines_so_far = [f for f in os.listdir(path) if ".csv" in f]
            fname = f"{self.method}_{self.dataset}_baseline{len(baselines_so_far)}.csv"



        file_path = os.path.join(path, fname)
        df.to_csv(file_path, index=True)






