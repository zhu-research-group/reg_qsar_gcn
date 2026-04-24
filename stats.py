from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score


def get_class_stats(model, X, y):
    """
    
    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return: 
    """
    if not model:
        predicted_probas = y
        predicted_classes = y.copy()
        predicted_classes[predicted_classes >= 0.5] = 1
        predicted_classes[predicted_classes < 0.5] = 0
        y = X
    else:
        if 'predict_classes' in dir(model):
            predicted_classes = model.predict_classes(X, verbose=0)[:, 0]
            predicted_probas = model.predict_proba(X, verbose=0)[:, 0]
        else:
            predicted_classes = model.predict(X)
            predicted_probas = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, predicted_classes)
    f1_sc = f1_score(y, predicted_classes)

    # Sometimes SVM spits out probabilties with of inf
    # so set them as 1
    from numpy import inf
    predicted_probas[predicted_probas == inf] = 1

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_tr, tpr_tr)
    # test classification results

    cohen_kappa = cohen_kappa_score(y, predicted_classes)
    matthews_corr = matthews_corrcoef(y, predicted_classes)
    precision = precision_score(y, predicted_classes)
    recall = recall_score(y, predicted_classes)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y, predicted_classes).ravel()
    specificity = tn / (tn + fp)
    ccr = (recall + specificity) / 2

    return {'ACC': acc, 'F1-Score': f1_sc, 'AUC': roc_auc, 'Cohen\'s Kappa': cohen_kappa,
            'MCC': matthews_corr, 'Precision/PPV': precision, 'Recall': recall, 'Specificity': specificity, 'CCR': ccr}


def get_regress_stats(model, X, y):
    """

    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return:
    """
    if not model:
        predictions = y
        y = X
    else:
        if 'predict_classes' in dir(model):
            predictions = model.predict(X, verbose=0)[:, 0]
        else:
            predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {'MAE': mae, 'r2': r2}

# scoring dictionary, just a dictionary containing the evaluation metrics passed through a make_scorer()
# fx, necessary for use in GridSearchCV

class_scoring = {'ACC': make_scorer(accuracy_score), 'F1-Score': make_scorer(f1_score),
                 'AUC': make_scorer(roc_auc_score),
                 'Cohen\'s Kappa': make_scorer(cohen_kappa_score), 'MCC': make_scorer(matthews_corrcoef),
                 'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score)}

regress_scoring = {'MAE': make_scorer(mean_absolute_error), 'r2_score': make_scorer(r2_score)}
