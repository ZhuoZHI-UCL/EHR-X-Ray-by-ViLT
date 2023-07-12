from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
def computeAUROC(y_true, predictions):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                        average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                        average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                            average="weighted")

    auprc = metrics.average_precision_score(y_true, predictions, average=None)

    
    auc_scores = []
    auprc_scores = []
    ci_auroc = []
    ci_auprc = []
    if len(y_true.shape) == 1:
        y_true = y_true[:, None]
        predictions = predictions[:, None]
    for i in range(y_true.shape[1]):
        df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': predictions[:, i]})
        (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc) = get_model_performance(df)
        auc_scores.append(test_auroc)
        auprc_scores.append(test_auprc)
        ci_auroc.append((lower_auroc, upper_auroc))
        ci_auprc.append((lower_auprc, upper_auprc))
    
    auc_scores = np.array(auc_scores)
    auprc_scores = np.array(auprc_scores)
    
    return { "auc_scores": auc_scores,
        
        "auroc_mean": np.mean(auc_scores),
        "auprc_mean": np.mean(auprc_scores),
        "auprc_scores": auprc_scores, 
        'ci_auroc': ci_auroc,
        'ci_auprc': ci_auprc,
        }



def auroc_myself(y_true,y_pre):
    try:
        auroc= roc_auc_score(y_true, y_pre,average='weighted')
        auprc= average_precision_score(y_true, y_pre,average='weighted')
    except ValueError:
        print("Warning: Only one class present in y_true. ROC AUC score is not defined in that case.")
        auroc = None
        auprc = None
    return (auroc)
