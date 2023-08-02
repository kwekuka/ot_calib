from sklearn.metrics import log_loss
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from dirichletcal.calib.matrixscaling import MatrixScaling
from dirichletcal.calib.vectorscaling import VectorScaling
from dirichletcal.calib.tempscaling import TemperatureScaling

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from wasscal.data import load_tt_split
from wasscal.logger import Logger



x_train, y_train, x_test, y_test = load_tt_split("pretrained_logits/densenet40_c10_logits.p", return_logits=False)
logger = Logger(dataset="cifar10",
                method="Densenet40",
                num_classes=x_test.shape[1])
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)



#Uncalibrated
logger.addEntry(
    test_probs=x_test,
    test_labels=y_test,
    train_probs=x_train,
    train_labels=y_train,
    method="uncal",
)
cla_loss = log_loss(y_test, x_test)
print("Uncalibrated: {:.4f}".format(cla_loss))

#Temperature Scaling
reg = []
calibrator = TemperatureScaling(logit_constant=0.0)
calibrator.fit(x_train, y_train)
cal_scores_test = calibrator.predict_proba(x_test)
cal_scores_train = calibrator.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Temp Scaling: {:.4f}".format(cal_loss))
logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ts",
)

#Vector Scaling
calibrator = VectorScaling(logit_constant=0.0)
calibrator.fit(x_train, y_train)
cal_scores_test = calibrator.predict_proba(x_test)
cal_scores_train = calibrator.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Vector Scaling: {:.4f}".format(cal_loss))

logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="vs",
)



#Full Dirichlet
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)

gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': [None]},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(x_train, y_train)
# print('Best parameters: {}'.format(gscv.best_params_))
cal_scores_test = gscv.predict_proba(x_test)
cal_scores_train = gscv.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Dirichlet: {:.4f}".format(cal_loss))

logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="dirichlet",
)


#ODIR Dirichlet
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=reg)
gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': reg},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(x_train, y_train)
# print('Best parameters: {}'.format(gscv.best_params_))
cal_scores_test = gscv.predict_proba(x_test)
cal_scores_train = gscv.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Dirichlet (ODIR): {:.4f}".format(cal_loss))

logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="odir_dirichlet",
)


#Matrix Scaling ODIR
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
calibrator = MatrixScaling(reg_lambda_list=reg, reg_mu_list=reg)
cgscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': reg},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(x_train, y_train)
# print('Best parameters: {}'.format(gscv.best_params_))
cal_scores_test = gscv.predict_proba(x_test)
cal_scores_train = gscv.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Matrix Scaling (ODIR): {:.4f}".format(cal_loss))

logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ms_odir",
)

#Matrix Scaling
calibrator = MatrixScaling()
calibrator.fit(x_train, y_train)
cal_scores_test = calibrator.predict_proba(x_test)
cal_scores_train = calibrator.predict_proba(x_train)
cal_loss = log_loss(y_test, cal_scores_test)
print("Matrix Scaling: {:.4f}".format(cal_loss))

logger.addEntry(
    test_probs=cal_scores_test,
    test_labels=y_test,
    train_probs=cal_scores_train,
    train_labels=y_train,
    method="ms",
)

logger.save(baseline=True)

