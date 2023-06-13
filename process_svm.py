from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix


labels = ['negative', 'neutral', 'positive']
test_Y = []
predict = []


def svm(train_x_arr, test_x_arr, train_Y, test_X, test_Y):
	model =  SVC(C=1, cache_size=300, class_weight=None, coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma=1, kernel='poly',
             max_iter=197, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False).fit(train_x_arr, train_Y)

	predictions_SVM_data = model.predict(test_x_arr)
	test_prediction_data = pd.DataFrame()
	test_prediction_data['text_clean'] = test_X
	test_prediction_data['new_label'] = predictions_SVM_data
	SVM_accuracy_data = accuracy_score(predictions_SVM_data, test_Y)*100
	# print(test_prediction_data)
	# SVM_accuracy_data
	return SVM_accuracy_data, predictions_SVM_data


def _report(TN, FP, FN, TP):
    TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
    TNR = TN/(TN+FP) if (TN+FP) != 0 else 0
    PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
    report = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
              'TPR': TPR, 'Recall': TPR, 'Sensitivity': TPR,
              'TNR': TNR, 'Specificity': TNR,
              'FPR': FP/(FP+TN) if (FP+TN) != 0 else 0,
              'FNR': FN/(FN+TP) if (FN+TP) != 0 else 0,
              'PPV': PPV, 'Precision': PPV,
              'F1 Score': 2*(PPV*TPR)/(PPV+TPR)
              }
    return report


def multi_classification_report(test_Y, predict, labels=None, encoded_labels=True, as_frame=False):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import multilabel_confusion_matrix

    conf_labels = None if encoded_labels else labels

    conf_mat = multilabel_confusion_matrix(test_Y, predict, labels=conf_labels)
    report = dict()
    if labels == None:
        counter = np.arange(len(conf_mat))
    else:
        counter = labels

    for i, name in enumerate(counter):
        TN, FP, FN, TP = conf_mat[i].ravel()
        report[name] = _report(TN, FP, FN, TP)

    if as_frame:
        return pd.DataFrame(report)
    return report


def summarized_classification_report(test_Y, predict, as_frame=False):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    report = dict()

    avg_list = ['micro', 'macro', 'weighted']
    for avg in avg_list:
        pre = precision_score(test_Y, predict, average=avg)
        re = recall_score(test_Y, predict, average=avg)
        f1 = f1_score(test_Y, predict, average=avg)
        report[avg] = {'Precision': pre,
                       'Recall': re,
                       'F1 Score': f1}
    if as_frame:
        return pd.DataFrame(report)
    return report


def svmProcess(train_x_arr, test_x_arr, train_Y, test_X, test_Y):
    accuracy, predict = svm(train_x_arr, test_x_arr, train_Y, test_X, test_Y)
    classification_report = multi_classification_report(test_Y, predict, labels=[
                                'negative', 'neutral', 'positive'], encoded_labels=True, as_frame=True)
    summarized_report = summarized_classification_report(test_Y, predict, as_frame=True)
    confusion_matrix_report = confusion_matrix(test_Y, predict)
    return accuracy, classification_report, summarized_report, confusion_matrix_report
    
