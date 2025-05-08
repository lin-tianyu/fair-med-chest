import numpy as np
import pandas as pd
import pickle
import os
from scipy.special import expit as sigmoid
from sklearn.metrics import log_loss, confusion_matrix, roc_auc_score
from typing import List, Union, Dict, Any, Tuple

# Metrics evaluation functions
def compute_mmpf_metrics(
    y_true: Union[List[int], np.ndarray],
    logits: Union[List[float], np.ndarray],
    sensitive_attr: Union[List[str], np.ndarray] = None,
    eps: float = 1e-15
) -> float:
    """
    Calculate risk (log loss) for a group
    
    Parameters:
        y_true: True labels (0/1)
        logits: Model raw logits
        sensitive_attr: Not used in this version - kept for API consistency
        eps: Smoothing value to prevent log_loss errors
    
    Returns:
        risk: The calculated log loss risk
    """
    y_true = np.array(y_true)
    logits = np.array(logits)
    
    # Calculate prediction probabilities
    y_pred = sigmoid(logits) if logits is not None else None
    
    if y_pred is None:
        return 0.0
    
    # Handle single-class case by adding a dummy example
    if len(np.unique(y_true)) == 1:
        y_true = np.append(y_true, 1 - y_true[0])
        y_pred = np.append(y_pred, 1 - y_pred[0])
        y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss (risk)
    risk = log_loss(y_true, y_pred)
    
    return risk

def compute_multiaccuracy(pred_probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiaccuracy (absolute residual mean)"""
    residual = pred_probs - labels
    return np.abs(np.mean(residual))

def compute_group_multiaccuracy(pred_probs: np.ndarray, labels: np.ndarray, group_indices: np.ndarray) -> float:
    """Compute multiaccuracy for specific subgroup (by indexing)"""
    if len(group_indices) == 0:
        return 0.0
    return compute_multiaccuracy(pred_probs[group_indices], labels[group_indices])

def expected_calibration_error(
    pred_probs: np.ndarray, 
    labels: np.ndarray, 
    num_bins: int = 10, 
    metric_variant: str = "abs", 
    quantile_bins: bool = False
) -> float:
    """
    Computes the calibration error with a binning estimator over equal sized bins
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut
    
    try:
        bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
        df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
        
        ece_df = (
            df.groupby("bin_id")
            .agg(
                pred_probs_mean=("pred_probs", "mean"),
                labels_mean=("labels", "mean"),
                bin_size=("pred_probs", "size"),
            )
            .assign(
                bin_weight=lambda x: x.bin_size / df.shape[0],
                err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
            )
        )
        result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
        if metric_variant == "rmse":
            result = np.sqrt(result)
    except:
        # Handle cases where binning fails (e.g., not enough unique values)
        result = 0.0
        
    return result

def binary_classification_report(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Generate metrics for binary classification"""
    metrics = {}
    
    # Handle edge cases where all predictions or labels are the same
    if len(np.unique(y)) < 2:
        # Can't calculate AUC with only one class
        metrics["auc"] = 0.5
    else:
        metrics["auc"] = roc_auc_score(y, pred)
    
    metrics["ece"] = expected_calibration_error(pred, y)
    metrics["ae"] = compute_multiaccuracy(pred, y)

    # Create binary predictions using 0.5 threshold
    binary_preds = (pred > 0.5).astype(int)
    
    # Calculate confusion matrix metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y, binary_preds).ravel()
        metrics.update({
            "acc": (tp + tn) / (tn + fp + fn + tp),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "tnr": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })
    except ValueError:
        # Handle case where confusion matrix fails (e.g., all predictions are one class)
        metrics.update({
            "acc": np.mean(y == binary_preds),
            "tpr": 0,
            "tnr": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        })
    
    return metrics

def evaluate_binary(
    pred: np.ndarray, 
    Y: np.ndarray, 
    A: np.ndarray, 
    logits: np.ndarray = None
) -> Tuple[Dict[str, float], Dict[str, List]]:
    """Evaluate binary classifier with fairness metrics across subgroups"""
    overall_metrics = {}
    subgroup_metrics = {}

    # Overall metrics
    overall_metrics.update(binary_classification_report(pred, Y))

    # Subgroup metrics
    for group in np.unique(A):
        group_indices = np.where(A == group)[0]
        sub_report = binary_classification_report(pred[group_indices], Y[group_indices])

        # Compute multiaccuracy using index-based filtering
        multiacc = compute_group_multiaccuracy(pred, Y, group_indices)
        sub_report["ae"] = multiacc
        
        # Compute MMPF metrics if logits are available
        if logits is not None:
            # Calculate risk directly for this group
            group_risk = compute_mmpf_metrics(Y[group_indices], logits[group_indices])
            sub_report["group_risk"] = group_risk
        else:
            sub_report["group_risk"] = 0.0

        for k, v in sub_report.items():
            subgroup_metrics.setdefault(k, []).append(v)

    return overall_metrics, subgroup_metrics

def organize_results(
    overall_metrics: Dict[str, float], 
    subgroup_metrics: Dict[str, List], 
    sensitive_attr: str = "sex"
) -> Dict[str, float]:
    """Organize metrics into a structured result dictionary"""
    # Get values from subgroup metrics
    subgroup_auc = subgroup_metrics["auc"]
    subgroup_acc = subgroup_metrics["acc"]
    subgroup_ece = subgroup_metrics["ece"]
    subgroup_tpr = subgroup_metrics["tpr"]
    subgroup_tnr = subgroup_metrics["tnr"]
    subgroup_multiacc = subgroup_metrics["ae"]
    
    # Extract group risks (now directly as a list of floats)
    if "group_risk" in subgroup_metrics:
        group_risk_values = subgroup_metrics['group_risk']
    else:
        group_risk_values = [0.0] * len(subgroup_auc)
    
    # Extract the metrics for each subgroup based on sensitive attribute
    female_metrics = {}
    male_metrics = {}
    young_metrics = {}
    old_metrics = {}
    
    # If there are two subgroups, assume they are female/male or young/old
    if len(subgroup_auc) >= 2:
        if sensitive_attr == "sex":
            female_metrics = {
                "female_auc": subgroup_auc[0],
                "female_acc": subgroup_acc[0],
                "female_ae": subgroup_multiacc[0],
                "female_ece": subgroup_ece[0],
                "female_tpr": subgroup_tpr[0],
                "female_risk": group_risk_values[0] if len(group_risk_values) > 0 else 0.0
            }
            male_metrics = {
                "male_auc": subgroup_auc[1],
                "male_acc": subgroup_acc[1],
                "male_ae": subgroup_multiacc[1],
                "male_ece": subgroup_ece[1],
                "male_tpr": subgroup_tpr[1],
                "male_risk": group_risk_values[1] if len(group_risk_values) > 1 else 0.0
            }
        else:  # age
            young_metrics = {
                "young_auc": subgroup_auc[0],
                "young_acc": subgroup_acc[0],
                "young_ae": subgroup_multiacc[0],
                "young_ece": subgroup_ece[0],
                "young_tpr": subgroup_tpr[0],
                "young_risk": group_risk_values[0] if len(group_risk_values) > 0 else 0.0
            }
            old_metrics = {
                "old_auc": subgroup_auc[1],
                "old_acc": subgroup_acc[1],
                "old_ae": subgroup_multiacc[1],
                "old_ece": subgroup_ece[1],
                "old_tpr": subgroup_tpr[1],
                "old_risk": group_risk_values[1] if len(group_risk_values) > 1 else 0.0
            }
    
    # Create final results dictionary
    result = {
        "overall-auc": overall_metrics["auc"],
        "overall-acc": overall_metrics["acc"],
        "worst-auc": min(subgroup_auc) if subgroup_auc else 0,
        "auc-gap": max(subgroup_auc) - min(subgroup_auc) if subgroup_auc else 0,
        "acc-gap": max(subgroup_acc) - min(subgroup_acc) if subgroup_acc else 0,
        "ae-gap": max(subgroup_multiacc) - min(subgroup_multiacc) if subgroup_multiacc else 0,
        "ece-gap": max(subgroup_ece) - min(subgroup_ece) if subgroup_ece else 0,
        "eod": 1 - ((max(subgroup_tpr) - min(subgroup_tpr)) + (max(subgroup_tnr) - min(subgroup_tnr))) / 2 if subgroup_tpr and subgroup_tnr else 0,
        "eo": max(subgroup_tpr) - min(subgroup_tpr) if subgroup_tpr else 0,
        "worst_group_risk": max(group_risk_values) if group_risk_values else 0,
        "risk-gap": max(group_risk_values) - min(group_risk_values) if len(group_risk_values) > 1 else 0,
        "group_mean_risk": np.mean(group_risk_values) if group_risk_values else 0,
    }
    
    # Add subgroup-specific metrics
    result.update(female_metrics)
    result.update(male_metrics)
    result.update(young_metrics)
    result.update(old_metrics)
    
    return result

# Data loading functions
def load_model_predictions(datapath: str, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load model predictions from pickle file"""
    model_path = f"{datapath}/predictions_{model_name}.pkl"
    
    if not os.path.exists(model_path):
        return None, None, None, None
        
    with open(model_path, 'rb') as file:
        predictions = pickle.load(file)
    
    probs = predictions.get('probs', None)
    labels = predictions.get('label', None)
    logits = predictions.get('logits', None)
    patientid = predictions.get('patientid', None)
    
    # Check if tensor and convert to regular list/array if needed
    if patientid is not None and len(patientid) > 0 and hasattr(patientid[0], 'item'):
        patientid = [pid.item() for pid in patientid]
    
    return probs, logits, labels, patientid

def load_dataset_metadata(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load metadata for a dataset"""
    # base_path = "/Users/carol/Desktop/DS_BME/project/fair-med-chest/metriceval/no_finding"
    base_path = "/Users/jerrylcj/python_proj/fair-med-chest/metriceval/no_finding"
    
    if dataset == "CXP":
        metadata_path = f"{base_path}/CXP/test_with_metadata.csv"
        metadata = pd.read_csv(metadata_path)
        patient_id_col = 'Patient'
        age_col = 'Age'
        sex_col = 'Sex'
    elif dataset == "MIMIC":
        metadata_path = f"{base_path}/MIMIC/test.csv"
        metadata = pd.read_csv(metadata_path)
        patient_id_col = 'subject_id'
        age_col = 'anchor_age'
        sex_col = 'gender'
    elif dataset == "NIH":
        metadata_path = f"{base_path}/NIH/test_meta_FM.csv"
        metadata = pd.read_csv(metadata_path)
        patient_id_col = 'patientid'
        age_col = 'Patient Age'
        sex_col = 'gender'
    else:
        return None, None, None
    
    # Extract relevant columns
    patient_ids = metadata[patient_id_col].values
    age = metadata[age_col].values
    sex = metadata[sex_col].values
    
    return patient_ids, sex, age

# Main calculation functions
def calculate_dataset_results(dataset: str, sensitive_attr: str = "sex", age_threshold: int = 65) -> pd.DataFrame:
    """Calculate fairness metrics for all models on a specific dataset"""
    # datapath = f"/Users/carol/Desktop/DS_BME/project/fair-med-chest/metriceval/no_finding/{dataset}"
    datapath = f"/Users/jerrylcj/python_proj/fair-med-chest/metriceval/no_finding/{dataset}"
    models = ["CLIP", "BiomedCLIP", "MedCLIP", "PubMedCLIP"]
    
    # Load metadata
    patient_ids, sex, age_raw = load_dataset_metadata(dataset)
    if patient_ids is None:
        return pd.DataFrame()
    
    # Create age binary with configurable threshold
    age_binary = (np.array(age_raw) >= age_threshold).astype(int)
    
    # Create sensitive group arrays
    sensitive_groups = {
        'sex': sex,
        'age': age_binary
    }
    
    # Create DataFrame for results
    result_df = pd.DataFrame()
    
    # Process each model
    for model_name in models:
        # Load model predictions
        probs, logits, labels, patientid = load_model_predictions(datapath, model_name)
        if probs is None or labels is None:
            continue
            
        # Select the sensitive attribute
        sensitive_attr_values = sensitive_groups[sensitive_attr]
        sen_dict = dict(zip(patient_ids, sensitive_attr_values))
        
        # Map sensitive attributes to patient IDs in the prediction data
        try:
            # Use the intersection of IDs to process only the data we have metadata for
            common_ids = set(patient_ids).intersection(set(patientid))
            if len(common_ids) == 0:
                print(f"No common patient IDs found for {model_name} in {dataset}")
                continue
                
            # Filter data to only include common IDs
            keep_indices = [i for i, pid in enumerate(patientid) if pid in common_ids]
            filtered_probs = probs[keep_indices]
            filtered_labels = np.array(labels)[keep_indices]
            filtered_patientid = np.array(patientid)[keep_indices]
            filtered_logits = logits[keep_indices] if logits is not None else None
            
            # Create sensitive attribute array for filtered data
            sen = np.array([sen_dict[pid] for pid in filtered_patientid])
            
            # Evaluate binary classifier
            if filtered_logits is not None:
                overall_metrics, subgroup_metrics = evaluate_binary(
                    pred=filtered_probs[:, 1], Y=filtered_labels, A=sen, logits=filtered_logits[:, 1] - filtered_logits[:, 0]
                )
            else:
                overall_metrics, subgroup_metrics = evaluate_binary(
                    pred=filtered_probs[:, 1], Y=filtered_labels, A=sen
                )
                
            # Organize results
            result = organize_results(overall_metrics, subgroup_metrics, sensitive_attr)
            
            # Store in the result DataFrame
            model_df = pd.DataFrame(result, index=[model_name])
            result_df = pd.concat([result_df, model_df])
        except Exception as e:
            print(f"Error processing {model_name} for {dataset}: {str(e)}")
            continue
    
    # Set the dataset name as an attribute
    result_df.name = dataset
    return result_df

def calculate_model_results(model: str, sensitive_attr: str = "sex", age_threshold: int = 65) -> pd.DataFrame:
    """Calculate fairness metrics for a specific model across all datasets"""
    datasets = ["CXP", "MIMIC", "NIH"]
    results = {}
    
    for dataset in datasets:
        # datapath = f"/Users/carol/Desktop/DS_BME/project/fair-med-chest/metriceval/no_finding/{dataset}"
        datapath = f"/Users/jerrylcj/python_proj/fair-med-chest/metriceval/no_finding/{dataset}"
        
        # Load metadata
        patient_ids, sex, age_raw = load_dataset_metadata(dataset)
        if patient_ids is None:
            continue
        
        # Create age binary with configurable threshold
        age_binary = (np.array(age_raw) >= age_threshold).astype(int)
        
        # Create sensitive group arrays
        sensitive_groups = {
            'sex': sex,
            'age': age_binary
        }
        
        # Load model predictions
        probs, logits, labels, patientid = load_model_predictions(datapath, model)
        if probs is None or labels is None:
            continue
            
        # Select the sensitive attribute
        sensitive_attr_values = sensitive_groups[sensitive_attr]
        sen_dict = dict(zip(patient_ids, sensitive_attr_values))
        
        # Map sensitive attributes to patient IDs in the prediction data
        try:
            # Use the intersection of IDs to process only the data we have metadata for
            common_ids = set(patient_ids).intersection(set(patientid))
            if len(common_ids) == 0:
                print(f"No common patient IDs found for {model} in {dataset}")
                continue
                
            # Filter data to only include common IDs
            keep_indices = [i for i, pid in enumerate(patientid) if pid in common_ids]
            filtered_probs = probs[keep_indices]
            filtered_labels = np.array(labels)[keep_indices]
            filtered_patientid = np.array(patientid)[keep_indices]
            filtered_logits = logits[keep_indices] if logits is not None else None
            
            # Create sensitive attribute array for filtered data
            sen = np.array([sen_dict[pid] for pid in filtered_patientid])
            
            # Evaluate binary classifier
            if filtered_logits is not None:
                overall_metrics, subgroup_metrics = evaluate_binary(
                    pred=filtered_probs[:, 1], Y=filtered_labels, A=sen, logits=filtered_logits[:, 1] - filtered_logits[:, 0]
                )
            else:
                overall_metrics, subgroup_metrics = evaluate_binary(
                    pred=filtered_probs[:, 1], Y=filtered_labels, A=sen
                )
                
            # Organize results
            result = organize_results(overall_metrics, subgroup_metrics, sensitive_attr)
            results[dataset] = result
        except Exception as e:
            print(f"Error processing {model} for {dataset}: {str(e)}")
            continue
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results).T
        results_df.name = model  # Add model name as attribute
        return results_df
    else:
        return pd.DataFrame()