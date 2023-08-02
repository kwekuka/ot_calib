from sklearn.metrics import log_loss
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from dirichletcal.calib.matrixscaling import MatrixScaling
from dirichletcal.calib.vectorscaling import VectorScaling
from dirichletcal.calib.tempscaling import TemperatureScaling

from sklearn.model_selection import (
                                StratifiedKFold,
                                GridSearchCV)
from wasscal.data import load_tt_split
from wasscal.logger import Logger



x_train, y_train, x_test, y_test = load_tt_split("pretrained_logits/densenet40_c10_logits.p", return_logits=False)
logger = Logger(dataset="cifar10",
                method="Densenet40",
                num_classes=x_test.shape[1])
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)



#Uncalibrated
cla_scores_train = x_train
cla_scores_test = x_test

cla_scores_train = x_train
cla_scores_test = x_test
logger.addEntry(
    test_probs=cla_scores_test,
    test_labels=y_test,
    train_probs=cla_scores_train,
    train_labels=y_train,
    method="uncal",
)

#Temperature Scaling
reg = []
calibrator = TemperatureScaling(logit_constant=0.0)
calibrator.fit(cla_scores_train, y_train)
cal_scores_test = calibrator.predict_proba(cla_scores_test)
cal_scores_train = calibrator.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ts",
)

#Vector Scaling
cla_scores_train = x_train
reg = []
calibrator = VectorScaling(logit_constant=0.0)
calibrator.fit(cla_scores_train, y_train)
cla_scores_test = x_test
cal_scores_test = calibrator.predict_proba(cla_scores_test)
cal_scores_train = calibrator.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="vs",
)



#Full Dirichlet
cla_scores_train = x_train
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)

gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': [None]},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(cla_scores_train, y_train)
print('Best parameters: {}'.format(gscv.best_params_))
cla_scores_test = x_test
cal_scores_test = gscv.predict_proba(cla_scores_test)
cal_scores_train = gscv.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="dirichlet",
)


#ODIR Dirichlet
cla_scores_train = x_train
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=reg)
gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': reg},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(cla_scores_train, y_train)
print('Best parameters: {}'.format(gscv.best_params_))
cla_scores_test = x_test
cal_scores_test = gscv.predict_proba(cla_scores_test)
cal_scores_train = gscv.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="odir_dirichlet",
)


#Matrix Scaling ODIR
cla_scores_train = x_train
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = MatrixScaling(reg_lambda_list=reg, reg_mu_list=reg)
calibrator.fit(cla_scores_train, y_train)

cla_scores_test = x_test
cal_scores_test = gscv.predict_proba(cla_scores_test)
cal_scores_train = gscv.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ms_odir",
)

#Matrix Scaling
cla_scores_train = x_train
reg = []
calibrator = MatrixScaling()
calibrator.fit(cla_scores_train, y_train)
cla_scores_test = x_test
cal_scores_test = calibrator.predict_proba(cla_scores_test)
cal_scores_train = calibrator.predict_proba(cla_scores_train)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.4f}, calibrator {:.4f}".format(
    cla_loss, cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ms",
)

logger.save(baseline=True)

