import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import sys
import os
import traceback
import time

# Import the metrics evaluation module
sys.path.append(os.path.dirname(__file__))
from metric_eval import calculate_dataset_results, calculate_model_results

# Constants and settings
MODELS = ["CLIP", "BiomedCLIP", "MedCLIP", "PubMedCLIP"]
DATASETS = ["CXP", "MIMIC", "NIH"]
SENSITIVE_ATTRS = ["sex", "age"]
FAIRNESS_METRICS = ["ae-gap", "ece-gap", "eod", "eo", "risk-gap"]
UTILITY_METRICS = ["overall-auc", "overall-acc"]

# Plot generation functions
def generate_fairness_bar_plot(results_df, mode="dataset"):
    """Generate bar plots for fairness metrics"""
    fig = Figure(figsize=(15, 10))
    axes = fig.subplots(2, 3)
    axes = axes.flatten()
    
    # Create 5 bar plots for each fairness metric
    for i, metric in enumerate(FAIRNESS_METRICS):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if mode == "dataset":
            # Different models for a selected dataset
            values = results_df[metric].values
            labels = results_df.index.tolist()  # Model names
            bar_colors = plt.cm.tab10(np.arange(len(labels)) % 10)
        else:  # mode == "model"
            # Different datasets for a selected model
            values = results_df[metric].values
            labels = results_df.index.tolist()  # Dataset names
            bar_colors = plt.cm.tab10(np.arange(len(labels)) % 10)
        
        bars = ax.bar(labels, values, color=bar_colors)
        ax.set_title(f"{metric}")
        
        if len(values) > 0:
            ax.set_ylim(0, min(1, max(values) * 1.5))
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        '''for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)'''
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        handles = [plt.Line2D([], [], marker='s', linestyle='', color=plt.cm.tab10(i), markersize=15)
            for i in range(len(labels))]
        axes[j].legend(handles, labels,
                title="Models" if mode=='dataset' else "Datasets",
                loc='center', ncol=1, frameon=False,fontsize=16, title_fontsize=20)
    
    if mode == "dataset":
        title = f"Fairness Metrics for Different Models on {results_df.name}"
    else:  # mode == "model"
        title = f"Fairness Metrics for {results_df.name} on Different Datasets"
        
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig

def generate_subgroup_comparison(results_df, sensitive_attr="sex", mode="dataset"):
    """Generate scatter plots comparing metrics across sensitive subgroups, with per-label colors and a legend in the 6th cell."""
    fig = Figure(figsize=(15, 10))
    axes = fig.subplots(2, 3)
    axes = axes.flatten()
    
    metrics = ["auc", "acc", "ae", "ece", "risk"]
    
    # common labels & colors
    labels = results_df.index.tolist()
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if sensitive_attr == "sex":
            x_group, y_group = "female", "male"
        else:
            x_group, y_group = "young", "old"
        
        try:
            x_values = results_df[f"{x_group}_{metric}"].values
            y_values = results_df[f"{y_group}_{metric}"].values
            
            # scatter with per-point colors
            for j, lab in enumerate(labels):
                ax.scatter(x_values[j], y_values[j], label=lab,
                           color=colors[j], alpha=0.7, s=100)
                ax.annotate(lab, (x_values[j], y_values[j]),
                            xytext=(5,5), textcoords='offset points')
            
            # y=x reference
            mn = min(min(x_values), min(y_values)) * 0.9
            mx = max(max(x_values), max(y_values)) * 1.1
            ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
            
            ax.set_xlabel(f"{x_group} {metric}")
            ax.set_ylabel(f"{y_group} {metric}")
            ax.set_title(f"{metric.upper()} Comparison")
        
        except Exception:
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
            ax.set_title(f"{metric.upper()} Comparison")
    
    # put legend in the last (6th) subplot
    legend_ax = axes[len(metrics)]
    legend_ax.axis('off')
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=colors[k], markersize=15)
        for k in range(len(labels))
    ]
    legend_ax.legend(handles, labels,
                     title="Models" if mode=='dataset' else "Datasets",
                     loc='center', frameon=False, fontsize=16, title_fontsize=20)
    
    title = (
        f"Subgroup Comparison ({sensitive_attr}) for Different Models on {results_df.name}"
        if mode=="dataset" else
        f"Subgroup Comparison ({sensitive_attr}) for {results_df.name} on Different Datasets"
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def generate_tradeoff_plots(results_df, mode="dataset"):
    """Generate utility–fairness tradeoff scatter plots, with per-label colors and a legend in the 6th cell."""
    fig = Figure(figsize=(15, 10))
    axes = fig.subplots(2, 3)
    axes = axes.flatten()
    
    # early exit on bad data
    if results_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        fig.suptitle("No Data Available", fontsize=16)
        fig.tight_layout()
        return fig
    
    required = FAIRNESS_METRICS + ["overall-auc"]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        for ax in axes:
            ax.text(0.5, 0.5, f"Missing columns: {', '.join(missing)}", ha='center', va='center')
        fig.suptitle("Data Format Error", fontsize=16)
        fig.tight_layout()
        return fig
    
    labels = results_df.index.tolist()
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    
    for i, metric in enumerate(FAIRNESS_METRICS):
        ax = axes[i]
        x = results_df[metric].fillna(0).values
        y = results_df["overall-auc"].fillna(0).values
        
        if len(x)==0 or len(y)==0:
            ax.text(0.5, 0.5, "No valid data points", ha='center', va='center')
            ax.set_title(f"Utility vs. {metric}")
            continue
        
        # Set the limits slightly expanded
        x_min = max(0, min(x) - 0.05)
        x_max = max(x) + 0.05
        y_min = max(0, min(y) - 0.05)
        y_max = min(1.0, max(y) + 0.05)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Create gradient background
        gradient_x = np.linspace(0, 1, 100)
        gradient_y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(gradient_x, gradient_y)
        
        # Create a gradient based on the metric
        if metric == "eod":
            # For EOD, higher is better, so upper-right (high EOD, high AUC) is best
            Z = (1-X) + (1-Y)  # This makes upper-right corner dark, lower-left light
            gradient_text = "Darker = Better Tradeoff (Upper-Right)"
        else:
            # For other fairness metrics, lower is better, so upper-left (low gap, high AUC) is best
            Z = X + (1-Y)  # This makes upper-left dark, lower-right light
            gradient_text = "Darker = Better Tradeoff (Upper-Left)"
        
        # Plot the gradient background
        gradient = ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], 
                           aspect='auto', alpha=0.2, cmap='Blues_r',
                           origin='lower', zorder=0)
        
        # Add a small note explaining the gradient
        ax.text(0.05, 0.05, gradient_text, 
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # per-point scatter (on top of gradient)
        for j, lab in enumerate(labels):
            ax.scatter(x[j], y[j], label=lab,
                       color=colors[j], alpha=0.7, s=100, zorder=5)
            ax.annotate(lab, (x[j], y[j]),
                        xytext=(5,5), textcoords='offset points', 
                        zorder=6)
        
        ax.set_xlabel(metric)
        ax.set_ylabel("AUC (Utility)")
        ax.set_title(f"Utility vs. {metric}")
        ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    
    # legend cell
    legend_ax = axes[len(FAIRNESS_METRICS)]
    legend_ax.axis('off')
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=colors[k], markersize=15)
        for k in range(len(labels))
    ]
    legend_ax.legend(handles, labels,
                     title="Models" if mode=='dataset' else "Datasets",
                     loc='center', frameon=False, fontsize=16, title_fontsize=20)
    
    title = (
        f"Utility–Fairness Tradeoff for Different Models on {results_df.name}"
        if mode=="dataset" else
        f"Utility–Fairness Tradeoff for {results_df.name} on Different Datasets"
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig

# Main Gradio application
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Model Fairness Analysis")
        
        with gr.Tabs() as tabs:
            # Tab 1: Dataset-based Analysis (Models comparison)
            with gr.TabItem("Dataset Analysis"):
                with gr.Row():
                    # Left panel
                    with gr.Column(scale=1):
                        dataset_dropdown = gr.Dropdown(
                            choices=DATASETS, 
                            label="Dataset",
                            value=DATASETS[0]
                        )
                        
                        sensitive_attr_radio = gr.Radio(
                            choices=SENSITIVE_ATTRS, 
                            label="Sensitive Attribute", 
                            value=SENSITIVE_ATTRS[0]
                        )
                        
                        age_threshold_slider = gr.Slider(
                            minimum=18, 
                            maximum=90, 
                            value=65, 
                            step=1, 
                            label="Age Threshold (for age attribute)"
                        )
                        
                        dataset_analyze_button = gr.Button("Analyze")
                        dataset_status = gr.Textbox(label="Status", value="Ready")
                        dataset_progress = gr.Slider(minimum=0, maximum=100, value=0, label="Progress")
                    
                    # Right panel
                    with gr.Column(scale=2):
                        with gr.Accordion("Fairness Metrics", open=True):
                            fairness_plot = gr.Plot(format="png")
                        
                        with gr.Accordion("Subgroup Comparison", open=True):
                            subgroup_plot = gr.Plot(format="png")
                            
                        with gr.Accordion("Utility-Fairness Tradeoff", open=True):
                            tradeoff_plot = gr.Plot(format="png")
            
            # Tab 2: Model-based Analysis (Datasets comparison)
            with gr.TabItem("Model Analysis"):
                with gr.Row():
                    # Left panel
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=MODELS, 
                            label="Model",
                            value=MODELS[0]
                        )
                        
                        model_sensitive_attr_radio = gr.Radio(
                            choices=SENSITIVE_ATTRS, 
                            label="Sensitive Attribute", 
                            value=SENSITIVE_ATTRS[0]
                        )
                        
                        model_age_threshold_slider = gr.Slider(
                            minimum=18, 
                            maximum=90, 
                            value=65, 
                            step=1, 
                            label="Age Threshold (for age attribute)"
                        )
                        
                        model_analyze_button = gr.Button("Analyze")
                        model_status = gr.Textbox(label="Status", value="Ready")
                        model_progress = gr.Slider(minimum=0, maximum=100, value=0, label="Progress")
                    
                    # Right panel
                    with gr.Column(scale=2):
                        with gr.Accordion("Fairness Metrics", open=True):
                            model_fairness_plot = gr.Plot(format="png")
                        
                        with gr.Accordion("Subgroup Comparison", open=True):
                            model_subgroup_plot = gr.Plot(format="png")
                            
                        with gr.Accordion("Utility-Fairness Tradeoff", open=True):
                            model_tradeoff_plot = gr.Plot(format="png")
        
        # Event handlers for Dataset Analysis tab
        def analyze_dataset(dataset, sensitive_attr, age_threshold):
            try:
                # Update status and reset progress
                yield "Loading metadata...", 0, None, None, None
                time.sleep(0.5)  # Small delay to show progress

                # Calculate metrics using imported function with progress updates
                yield "Loading model predictions...", 20, None, None, None
                time.sleep(0.5)  # Small delay to show progress
                
                yield "Calculating metrics...", 40, None, None, None
                results_df = calculate_dataset_results(dataset, sensitive_attr, age_threshold)
                
                if results_df.empty:
                    return "No data available for selected dataset", 0, None, None, None
                
                # Generate plots
                yield "Generating fairness plots...", 60, None, None, None
                try:
                    fairness_fig = generate_fairness_bar_plot(results_df, mode="dataset")
                except Exception as e:
                    return f"Error in fairness plot: {str(e)}\n{traceback.format_exc()}", 0, None, None, None
                
                yield "Generating subgroup comparison plots...", 80, fairness_fig, None, None
                try:
                    subgroup_fig = generate_subgroup_comparison(results_df, sensitive_attr, mode="dataset")
                except Exception as e:
                    return f"Error in subgroup plot: {str(e)}\n{traceback.format_exc()}", 0, fairness_fig, None, None
                
                yield "Generating tradeoff plots...", 90, fairness_fig, subgroup_fig, None
                try:
                    tradeoff_fig = generate_tradeoff_plots(results_df, mode="dataset")
                except Exception as e:
                    return f"Error in tradeoff plot: {str(e)}\n{traceback.format_exc()}", 0, fairness_fig, subgroup_fig, None
                
                # Update status
                yield "Analysis complete", 100, fairness_fig, subgroup_fig, tradeoff_fig
                return
            
            except Exception as e:
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return error_msg, 0, None, None, None
        
        dataset_analyze_button.click(
            fn=analyze_dataset,
            inputs=[
                dataset_dropdown,
                sensitive_attr_radio,
                age_threshold_slider
            ],
            outputs=[
                dataset_status,
                dataset_progress,
                fairness_plot,
                subgroup_plot,
                tradeoff_plot
            ]
        )
        
        # Event handlers for Model Analysis tab
        def analyze_model(model, sensitive_attr, age_threshold):
            try:
                # Update status and reset progress
                yield "Loading dataset metadata...", 0, None, None, None
                time.sleep(0.5)  # Small delay to show progress

                # Calculate metrics using imported function
                yield "Loading model predictions...", 20, None, None, None
                time.sleep(0.5)  # Small delay to show progress
                
                yield "Calculating metrics...", 40, None, None, None
                results_df = calculate_model_results(model, sensitive_attr, age_threshold)
                
                if results_df.empty:
                    return "No data available for selected model", 0, None, None, None
                
                # Generate plots with error handling
                yield "Generating fairness plots...", 60, None, None, None
                try:
                    fairness_fig = generate_fairness_bar_plot(results_df, mode="model")
                except Exception as e:
                    return f"Error in fairness plot: {str(e)}\n{traceback.format_exc()}", 0, None, None, None
                
                yield "Generating subgroup comparison plots...", 80, fairness_fig, None, None
                try:
                    subgroup_fig = generate_subgroup_comparison(results_df, sensitive_attr, mode="model")
                except Exception as e:
                    return f"Error in subgroup plot: {str(e)}\n{traceback.format_exc()}", 0, fairness_fig, None, None
                
                yield "Generating tradeoff plots...", 90, fairness_fig, subgroup_fig, None
                try:
                    tradeoff_fig = generate_tradeoff_plots(results_df, mode="model")
                except Exception as e:
                    return f"Error in tradeoff plot: {str(e)}\n{traceback.format_exc()}", 0, fairness_fig, subgroup_fig, None
                
                # Update status
                yield "Analysis complete", 100, fairness_fig, subgroup_fig, tradeoff_fig
                return
            
            except Exception as e:
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return error_msg, 0, None, None, None
        
        model_analyze_button.click(
            fn=analyze_model,
            inputs=[
                model_dropdown,
                model_sensitive_attr_radio,
                model_age_threshold_slider
            ],
            outputs=[
                model_status,
                model_progress,
                model_fairness_plot,
                model_subgroup_plot,
                model_tradeoff_plot
            ]
        )
        
    return demo

# Launch the Gradio app
if __name__ == "__main__":
    app = create_interface()
    app.launch()