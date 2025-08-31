"""
This script is used for tuning, and training the simple ML classification models.
"""

import os
import sys
sys.path.append("transferLearnUtilities.py")
from transferLearnUtilities import prepare_tr_learn_data, print_breaker

import numpy as np
import pandas as pd

import optuna

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    roc_curve, accuracy_score, precision_recall_fscore_support

import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network  # Import MLPClassifier

import xgboost as xgb

import joblib
import matplotlib.pyplot as plt
import logging

formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')


def check_dir(path):
    """
    This function is used for creating directory at the specified path if it does not exist.
    :param path: path where the directory is to be created
    :return: None
    """
    os.makedirs(path, exist_ok=True)


def save_trained_model(logger, path, model, name=None):
    """
    This function is used to save the trained ML classifier.
    :param logger: logger object to add information to the log file
    :param path: path to save the model
    :param model: trained ML classifier (model object)
    :param name: name of the trained classifier
    :return: None
    """
    os.makedirs(path, exist_ok=True)

    model_path = os.path.join(path, name + ".model")
    # save the trained model
    joblib.dump(model, model_path)
    logger.info(f"trained model %s saved to location %s." % (name, model_path))


def load_trained_model(path):
    """
    This function is used for loading trained ML classifier
    :param path: path to the trained classifier
    :return: trained classifier object
    """
    print("loading the trained model from path %s " % path)
    trained_model = joblib.load(path)
    return trained_model



def get_testing_statistics(logger, model, x_test, y_test, stats_path=None):
    """
    This function is used to obtain all the statistics for the model.
    :param logger: logger object to add information to the log file
    :param model: model object for which the statistics are to be obtained.
    :param x_test: testing data matrix
    :param y_test: testing data labels
    :param stats_path: path to save the testing statistics
    :return: None
    """
    logger.info(f"getting statistics on the testing dataset ...")
    # Obtain model predictions on test set
    y_pred = model.predict(x_test)

    # Save x_test, y_test, and y_pred using NumPy
    np.save(os.path.join(stats_path, 'x_test.npy'), x_test)
    np.save(os.path.join(stats_path, 'y_test.npy'), y_test)
    np.save(os.path.join(stats_path, 'y_pred.npy'), y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Print evaluation metrics
    logger.info('Accuracy: %s', accuracy)
    logger.info('Precision: %s', precision)
    logger.info('Recall: %s', recall)
    logger.info('F1-score: %s', f1_score)
    logger.info('ROC-AUC score: %s', roc_auc)
    logger.info('True positives: %s', tp)
    logger.info('False positives: %s', fp)
    logger.info('True negatives: %s', tn)
    logger.info('False negatives: %s', fn)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    fig = plt.figure()
    plt.imshow(cm, cmap=plt.cm.Purples)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.text(0, 0, f'TN = {tn}', ha='center', va='center', color='black')
    plt.text(1, 0, f'FP = {fp}', ha='center', va='center', color='black')
    plt.text(0, 1, f'FN = {fn}', ha='center', va='center', color='black')
    plt.text(1, 1, f'TP = {tp}', ha='center', va='center', color='black')
    conf_matrix_fig_path = os.path.join(stats_path, "conf_matrix.png")
    plt.savefig(conf_matrix_fig_path, dpi=200)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot ROC curve
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    roc_auc_fig_path = os.path.join(stats_path, "roc-auc-plot.png")
    plt.savefig(roc_auc_fig_path, dpi=200)

    # Calculate additional metrics
    tpr = tp / (tp + fn)  # True Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate
    fpr = fp / (tn + fp)  # False Positive Rate
    fnr = fn / (tp + fn)  # False Negative Rate

    logger.info('True Positive Rate (TPR): %s', tpr)
    logger.info('True Negative Rate (TNR): %s', tnr)
    logger.info('False Positive Rate (FPR): %s', fpr)
    logger.info('False Negative Rate (FNR): %s', fnr)

    # Save evaluation metrics to CSV file
    csv_file_path = os.path.join(stats_path, "testing-stats.csv")
    if csv_file_path is not None:
        metrics = {
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1-score': [f1_score],
            'roc-auc': [roc_auc],
            'tp': [tp],
            'fp': [fp],
            'tn': [tn],
            'fn': [fn],
            'tpr': [tpr],
            'tnr': [tnr],
            'fpr': [fpr],
            'fnr': [fnr]
            }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(csv_file_path, index=False)

    logger.info(f"statistics of the testing dataset are saved at location %s." % csv_file_path)


def get_testing_scores(logger, model, x_test, y_test):
    """
    This function is used for obtaining testing scores for the testing dataset.
    :param logger: logger object to add information to the log file
    :param model: model for which testing is to be performed
    :param x_test: testing data matrix
    :param y_test: testing data labels
    :return: testing score
    """
    test_score = model.score(x_test, y_test)
    logger.info(f"test score: {test_score}")
    return test_score


def get_validation_score(logger, model, x_train, y_train, x_test, y_test):
    """
    This function is used for obtaining validation, and cross-validation score for the tuned parameters
    :param logger: logger object to add information to the log file
    :param model: model object for which validation, and cross-validation scores are to be obtained
    :param x_train: training data matrix
    :param y_train: training data labels
    :param x_test: validation data matrix
    :param y_test: validation data labels
    :return:
    """
    val_score = model.score(x_test, y_test)
    logger.info(f"testing score: {val_score}")
    logger.info(f"performing cross-validation ...")
    cross_valid_score = cross_val_score(model, x_train, y_train, n_jobs=-1, cv=5)
    logger.info(f"cross-validation scores: {cross_valid_score}")
    logger.info(f"average cross-validation score: {cross_valid_score.mean()}")
    return val_score


def load_data_classical_models(dataset_path, logger, n_samples_per_class, apply_pca=False):
    """
    This function is used for loading the dataset required for training the classical ML models.
    :param n_samples_per_class: Number of samples to be loaded per class
    :param apply_pca: Boolean variable to apply PCA to the dataset, if True the data will be transformed to the PCA
    space
    :param logger: logger object to add information to the log file
    :param dataset_path: path to the csv file containing the dataset.
    :return:
    """
    logger.info("loading dataset ...")
    # Point to the directory storing data
    logger.info("loading the dataset from %s" % dataset_path)

    # preparing the training, validation, and testing dataset for transfer learning
    print(f"preparing the training dataset ...")
    train_data_path = os.path.join(dataset_path, "train-data.npz")
    features, labels, features_labels_df = prepare_tr_learn_data(train_data_path)
    data_df = features_labels_df.groupby("label").head(n_samples_per_class)
    # print(features_labels_df.head())
    print_breaker("training dataset processed successfully!")

    logger.info("shape of the training dataset is : %s" % str(data_df.shape))
    # print(data_df.head())

    print(f"preparing the testing dataset ...")
    test_data_path = os.path.join(dataset_path, "test-data.npz")
    x_test, y_test, test_features_labels_df = prepare_tr_learn_data(test_data_path)
    test_data_df = test_features_labels_df
    # print(test_data_df.head())
    print_breaker("testing dataset processed successfully!")

    logger.info("shape of the testing dataset is : %s" % str(data_df.shape))

    # obtaining unique number of classes in the dataset
    nb_classes = len(np.unique(data_df["label"]))
    logger.info("number of unique classes in the dataset is: %s" % str(nb_classes))

    if apply_pca:
        logger.info("transforming the dataset to PCA space ...")
        features = data_df.iloc[:, 1:]
        standardized_data = pd.DataFrame(preprocessing.scale(features), columns=features.columns)
        # Create a PCA object with the number of components you want to keep
        pca = PCA(n_components=len(data_df.columns) - 1, svd_solver='auto')

        # Fit the PCA object to the standardized data
        pca.fit(standardized_data)
        logger.info("Explained variance = %s" % (str(pca.explained_variance_)))
        logger.info("Explained variance ratio = %s" % (str(pca.explained_variance_ratio_)))

        # Transform the data to its principal components
        transformed_data = pca.transform(standardized_data)
        column_names = ["pc-" + str(col) for col in range(len(data_df.columns) - 1)]
        features = pd.DataFrame(transformed_data, columns=column_names)
        # print(features.head())
        logger.info("shape of the training data is: %s" % str(features.shape))
    else:
        x_train = data_df.iloc[:, 1:]
        y_train = data_df["label"]

        x_test = test_data_df.iloc[:, 1:]
        y_test = test_data_df["label"]
        logger.info("shape of the training data is: %s" % str(features.shape))

    labels = data_df["label"]
    logger.info("number of labels are: %s" % str(labels.shape))

    x_train = x_train.to_numpy()
    # print(f"x_train: {x_train}")
    x_test = x_test.to_numpy()
    # print(f"x_test: {x_test}")

    logger.info("input size of the training data: %s" % (str(x_train.shape)))
    logger.info("input size of the testing data: %s" % (str(x_test.shape)))

    return x_train, y_train, x_test, y_test, nb_classes


# Generalized objective function
def classifier_objective(trial, trial_number, classifier_type, x_train, y_train, x_test, y_test, logger):
    """
    Generalized objective function for tuning parameters of various classifiers.
    :param trial: trials object for optuna
    :param trial_number: serial number of current trial
    :param classifier_type: Type of classifier to tune
    :param x_train, y_train: Training data and labels
    :param logger: logger object for logging
    :return: Validation score
    """

    logger.info(f"currently tuning trial {trial_number} ...")

    if classifier_type == 'XGBoost':
        params = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',  # Use GPU acceleration
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            }
        clf = xgb.XGBClassifier(**params, use_label_encoder=False)
        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False, early_stopping_rounds=100)
        val_score = get_validation_score(logger, clf, x_train, y_train, x_test, y_test)

    else:
        raise ValueError("Unsupported classifier type")

    logger.info("*"*79)
    return val_score


def tune_classifier(classifier_type, params, model_stats_path, n_samples_per_class, n_trials=100, apply_pca=None):
    """
    This function is used for tuning parameters of the specified classifier.
    :param classifier_type: type of the shallow-ml model to be trained
    :param params: parameters of the dataset, and path to save the model, logs, and stats
    :param model_stats_path: path to save the stats of the model
    :param n_samples_per_class: number of samples per class for training
    :param n_trials: number of trials for which the model is to be tuned
    :param apply_pca: True if PCA should be applied to the data else False
    :return: None
    """

    # Create a logger object for this function
    logger = logging.getLogger(classifier_type)
    logger.setLevel(logging.INFO)
    check_dir(model_stats_path)
    log_file_path = os.path.join(model_stats_path, classifier_type + "-logs.log")
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger.info("=" * 90)
    logger.info(f"tuning parameters for {classifier_type} classifier ...")

    x_train, y_train, x_test, y_test, nb_classes = \
        load_data_classical_models(params["dataset_path"], logger, n_samples_per_class, apply_pca=apply_pca)

    check_dir(model_stats_path)

    storage = os.path.join("sqlite:///", model_stats_path, classifier_type + "_classifier.db")
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                storage=storage,
                                direction="maximize",
                                study_name=f"tuning-{classifier_type}-classifier",
                                load_if_exists=True)

    study.optimize(lambda trial: classifier_objective(trial, trial.number, classifier_type, x_train, y_train,
                                                      x_test, y_test,
                                                                              logger), n_trials=n_trials, n_jobs=4)

    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"best params: {best_params}")
    logger.info(f"best validation score: {best_score}")

    if classifier_type == "XGBoost":
        # initialize classifier with hyper-parameters
        clf = xgb.XGBClassifier(**best_params, use_label_encoder=False, verbosity=0)
        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False, early_stopping_rounds=100)

    # save the trained model
    trained_clf_path = os.path.join(params["trained_models_path"], clf.__class__.__name__)
    save_trained_model(logger, trained_clf_path, clf, name=clf.__class__.__name__)

    get_testing_statistics(logger, clf, x_test, y_test, stats_path=trained_clf_path)
    test_score = get_testing_scores(logger, clf, x_test, y_test)

    logger.info(f"tuning of the parameters for {classifier_type} classifier completed.")
    logger.info("=" * 99)

    # close the logger
    logger.removeHandler(console_handler)
    console_handler.close()
