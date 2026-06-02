import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from typing import List, Tuple, Dict

def quadratic_weighted_kappa(y_true: List[float], y_pred: List[float], max_score: int = 100) -> float:
    """Calculate Quadratic Weighted Kappa for grading evaluation."""
    y_true_int = [int(round(x)) for x in y_true]
    y_pred_int = [int(round(x)) for x in y_pred]
    
    try:
        n = max_score + 1
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weights[i, j] = 1 - ((i - j) ** 2) / (n ** 2)
        
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(y_true_int, y_pred_int, labels=list(range(n)))
        
        hist_true = np.sum(conf_mat, axis=1)
        hist_pred = np.sum(conf_mat, axis=0)
        expected = np.outer(hist_true, hist_pred) / np.sum(hist_true)
        
        observed_agreement = np.sum(weights * conf_mat) / np.sum(conf_mat)
        expected_agreement = np.sum(weights * expected) / np.sum(expected)
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return float(kappa)
    except Exception as e:
        print(f"Kappa calculation error: {e}")
        return float(cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic'))

def root_mean_square_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate RMSE between ground truth and predictions"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_model(model_scores: List[float], ground_truth: List[float], 
                   variation_flags: List[bool] = None, max_score: int = 100) -> Dict:
    """Evaluate model performance for research questions RQ1 and RQ2."""
    results = {
        'qwk_all': quadratic_weighted_kappa(ground_truth, model_scores, max_score),
        'rmse_all': root_mean_square_error(ground_truth, model_scores),
        'total_samples': len(model_scores)
    }
    
    if variation_flags is not None:
        standard_indices = [i for i, flag in enumerate(variation_flags) if not flag]
        variation_indices = [i for i, flag in enumerate(variation_flags) if flag]
        
        if standard_indices:
            standard_true = [ground_truth[i] for i in standard_indices]
            standard_pred = [model_scores[i] for i in standard_indices]
            results['qwk_standard'] = quadratic_weighted_kappa(standard_true, standard_pred, max_score)
            results['rmse_standard'] = root_mean_square_error(standard_true, standard_pred)
            results['standard_samples'] = len(standard_indices)
        else:
            results['qwk_standard'] = None
            results['standard_samples'] = 0
            
        if variation_indices:
            variation_true = [ground_truth[i] for i in variation_indices]
            variation_pred = [model_scores[i] for i in variation_indices]
            results['qwk_variation'] = quadratic_weighted_kappa(variation_true, variation_pred, max_score)
            results['rmse_variation'] = root_mean_square_error(variation_true, variation_pred)
            results['variation_samples'] = len(variation_indices)
        else:
            results['qwk_variation'] = None
            results['variation_samples'] = 0
    else:
        results['qwk_standard'] = None
        results['qwk_variation'] = None
        results['rmse_standard'] = None
        results['rmse_variation'] = None
        results['standard_samples'] = 0
        results['variation_samples'] = 0
    
    return results
