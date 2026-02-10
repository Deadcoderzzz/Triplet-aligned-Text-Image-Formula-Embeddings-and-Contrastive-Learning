import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(labels, predictions):
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary', zero_division=0),
        'recall': recall_score(labels, predictions, average='binary', zero_division=0),
        'f1': f1_score(labels, predictions, average='binary', zero_division=0)
    }
    
    return metrics


def compute_retrieval_metrics(similarities, labels, k_values=[1, 5, 10]):
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    
    metrics = {}
    
    for k in k_values:
        # Count relevant items in top K
        top_k_labels = sorted_labels[:k]
        recall_at_k = np.sum(top_k_labels) / max(np.sum(labels), 1)
        metrics[f'recall@{k}'] = recall_at_k
    
    return metrics


def plot_confusion_matrix(labels, predictions, save_path=None):
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dissimilar', 'Similar'],
                yticklabels=['Dissimilar', 'Similar'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_recall_curve(similarities_list, labels_list, model_names, save_path=None):
    k_values = [1, 2, 3, 5, 10, 20, 30, 40, 50]
    
    plt.figure(figsize=(10, 6))
    
    for similarities, labels, name in zip(similarities_list, labels_list, model_names):
        recalls = []
        
        for k in k_values:
            metrics = compute_retrieval_metrics(similarities, labels, k_values=[k])
            recalls.append(metrics[f'recall@{k}'] * 100)  # Convert to percentage
        
        plt.plot(k_values, recalls, marker='o', label=name, linewidth=2)
    
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Recall@K (%)', fontsize=12)
    plt.title('Recall@K Curves on SciOL Benchmark', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Recall curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(labels, predictions):
    """Print detailed classification report"""
    report = classification_report(
        labels, predictions,
        target_names=['Dissimilar', 'Similar'],
        digits=4
    )
    print("\nClassification Report:")
    print(report)


def analyze_errors(labels, predictions, similarities, threshold=0.5):
    labels = np.array(labels)
    predictions = np.array(predictions)
    similarities = np.array(similarities)
    
    # False positives: predicted similar but actually dissimilar
    fp_indices = np.where((predictions == 1) & (labels == 0))[0]
    fp_similarities = similarities[fp_indices]
    
    # False negatives: predicted dissimilar but actually similar
    fn_indices = np.where((predictions == 0) & (labels == 1))[0]
    fn_similarities = similarities[fn_indices]
    
    analysis = {
        'num_false_positives': len(fp_indices),
        'num_false_negatives': len(fn_indices),
        'fp_avg_similarity': float(np.mean(fp_similarities)) if len(fp_similarities) > 0 else 0,
        'fn_avg_similarity': float(np.mean(fn_similarities)) if len(fn_similarities) > 0 else 0,
        'threshold': threshold
    }
    
    return analysis
