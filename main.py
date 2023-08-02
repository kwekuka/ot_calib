import jax
import optax
import numpy as np
import jax.numpy as jnp
from wasscal.model import L2Dual
from wasscal.logger import Logger
from ott.problems.nn import dataset as dset
from ott.solvers.nn import models, neuraldual
from sklearn.model_selection import train_test_split
from wasscal.data import Sampler, supervised_calibrate
from ott.tools import plot, sinkhorn_divergence
from ott.geometry import pointcloud


@jax.jit
def sinkhorn_loss(x, y, epsilon=0.001):
    """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, epsilon=epsilon, a=a, b=b
    )
    return sdiv.divergence
clf = "resnet110"
dataset = "cifar10-resnet110"
method = "wasscal"

eval_batch_size = 1000
train_batch_size = 2048
num_train_iters = 2000

#Generate Wasserstein Calibrated Files
probs, calibrated, labels = supervised_calibrate(file_path="pretrained_logits/resnet110_SD_c10_logits.p",
                                  folder=dataset,
                                  save=True,
                                  file_name=clf)

num_classes = probs.shape[1]

#Generate training, testing, and validation data
tt_split = 0.5
source_train, source_test, target_train, target_test, y_train, y_test = train_test_split(
    probs,
    calibrated,
    labels,
    shuffle=True,
    train_size=tt_split
)

source_test, source_valid, target_test, target_valid, y_test, y_valid = train_test_split(
    source_test,
    target_test,
    y_test,
    shuffle=False,
    train_size=0.5
)

#Create logger for saving results
logger = Logger(dataset=dataset,
                method=method,
                num_classes=probs.shape[1])

source_train_sampler = Sampler(source_train, batch_size=train_batch_size)
target_train_sampler = Sampler(target_train, batch_size=train_batch_size)

train_dataloader = dset.Dataset(
    source_iter=source_train_sampler,
    target_iter=target_train_sampler
)

source_valid_sampler=Sampler(source_valid, batch_size=train_batch_size)
target_valid_sampler=Sampler(target_valid, batch_size=train_batch_size)

valid_dataloader = dset.Dataset(
    source_iter=source_valid_sampler,
    target_iter=target_valid_sampler
)

# initialize models and optimizers
input_dim = source_train.shape[1]

neural_f = models.MLP(dim_hidden=[256]*3)
neural_g = models.MLP(dim_hidden=[256]*3)


lr_schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=num_train_iters, alpha=1e-3)
# lr_schedule = optax.constant_schedule(1e-2)

optimizer_f = optax.adamw(learning_rate=lr_schedule)
optimizer_g = optax.adamw(learning_rate=lr_schedule)

def training_callback(step, learned_potentials, log):
    log.addEntryJax(
        test_probs=source_test,
        test_labels=y_test,
        train_probs=source_train,
        train_labels=y_train,
        iteration=step,
        model=learned_potentials
    )
    pred_target = learned_potentials.transport(source_test)
    print(
    f"Sinkhorn distance: {sinkhorn_loss(pred_target, target_test):.6f}"
    )
    # print(f"MSE:{mean_squared_error(pred_target, source_test)}")

neural_dual_solver = neuraldual.W2NeuralDual(
    dim_data=num_classes,
    neural_f=neural_f,
    neural_g=neural_g,
    optimizer_f=optimizer_f,
    optimizer_g=optimizer_g,
    num_train_iters=num_train_iters
)
logger = Logger(dataset=dataset, method=method, num_classes=num_classes)
callback_lambda = lambda x, y: training_callback(x, y, log=logger)
learned_potentials = neural_dual_solver(
    *train_dataloader,
    *valid_dataloader,
    callback=callback_lambda
)

logger.save()
