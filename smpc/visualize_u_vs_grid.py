#!/usr/bin/env python3
"""
Interactive visualization of battery power (u) vs grid power
for all simulation timesteps.

Usage:
    python3 visualize_u_vs_grid.py [controller_type]

    controller_type: "mpc", "sdp", or "rule_based" (optional, defaults to config.yaml)
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_results(controller_type):
    """Load simulation results."""
    config = load_config()
    results_dir = Path(config['output']['results_directory'])

    # Try standard name first
    results_file = results_dir / f"results_{controller_type}.csv"

    # If not found, try with " (numba jit)" suffix for SDP
    if not results_file.exists() and controller_type == "sdp":
        results_file = results_dir / f"results_{controller_type} (numba jit).csv"

    # If not found, try with dash instead of underscore
    if not results_file.exists() and controller_type == "rule_based":
        results_file = results_dir / f"results_rule-based.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return pd.read_csv(results_file)

def create_interactive_plot(df, controller_type):
    """Create interactive plot with slider to select timestep."""

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.7, 0.3],
        column_widths=[0.5, 0.5],
        subplot_titles=(
            f'Battery Power (u) vs Grid Power',
            'Timestep Info',
            'Battery Power Time Series',
            'Grid Power Time Series'
        ),
        specs=[[{"type": "scatter"}, {"type": "table"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Main scatter plot: u vs grid_power
    # Color by timestep for visual reference
    scatter = go.Scatter(
        x=df['grid_power'],
        y=df['battery_power'],
        mode='markers',
        marker=dict(
            size=6,
            color=df.index,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Timestep",
                x=0.45,
                y=0.85,
                len=0.5
            ),
            line=dict(width=0.5, color='white')
        ),
        text=[f"Step {i}" for i in df.index],
        customdata=df[['timestamp', 'grid_power', 'battery_power', 'battery_soc', 'electricity_price']].values,
        hovertemplate='<b>Step %{text}</b><br>' +
                     'Time: %{customdata[0]}<br>' +
                     'Grid: %{customdata[1]:.3f} kW<br>' +
                     'Battery: %{customdata[2]:.3f} kW<br>' +
                     'SOC: %{customdata[3]:.3f}<br>' +
                     'Price: %{customdata[4]:.4f} €/kWh<br>' +
                     '<extra></extra>',
        name='All timesteps',
        showlegend=False
    )
    fig.add_trace(scatter, row=1, col=1)

    # Add highlighted point (initially at first timestep)
    highlight = go.Scatter(
        x=[df['grid_power'].iloc[0]],
        y=[df['battery_power'].iloc[0]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='star',
            line=dict(width=2, color='darkred')
        ),
        customdata=[df[['timestamp', 'grid_power', 'battery_power', 'battery_soc', 'electricity_price']].iloc[0].values],
        hovertemplate='<b>SELECTED Step 0</b><br>' +
                     'Time: %{customdata[0][0]}<br>' +
                     'Grid: %{customdata[0][1]:.3f} kW<br>' +
                     'Battery: %{customdata[0][2]:.3f} kW<br>' +
                     'SOC: %{customdata[0][3]:.3f}<br>' +
                     'Price: %{customdata[0][4]:.4f} €/kWh<br>' +
                     '<extra></extra>',
        name='Selected',
        showlegend=False
    )
    fig.add_trace(highlight, row=1, col=1)

    # Info table
    info_table = go.Table(
        header=dict(
            values=['<b>Property</b>', '<b>Value</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                ['Timestep', 'Time', 'Grid Power', 'Battery Power', 'SOC', 'Price'],
                [
                    '0',
                    df['timestamp'].iloc[0],
                    f"{df['grid_power'].iloc[0]:.3f} kW",
                    f"{df['battery_power'].iloc[0]:.3f} kW",
                    f"{df['battery_soc'].iloc[0]:.3f}",
                    f"{df['electricity_price'].iloc[0]:.4f} €/kWh"
                ]
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )
    fig.add_trace(info_table, row=1, col=2)

    # Time series: battery_power
    ts_battery = go.Scatter(
        x=df.index,
        y=df['battery_power'],
        mode='lines',
        name='Battery Power',
        line=dict(color='blue', width=1.5),
        showlegend=False
    )
    fig.add_trace(ts_battery, row=2, col=1)

    # Vertical line for selected timestep
    vline_battery = go.Scatter(
        x=[0, 0],
        y=[df['battery_power'].min(), df['battery_power'].max()],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    )
    fig.add_trace(vline_battery, row=2, col=1)

    # Time series: grid_power
    ts_grid = go.Scatter(
        x=df.index,
        y=df['grid_power'],
        mode='lines',
        name='Grid Power',
        line=dict(color='orange', width=1.5),
        showlegend=False
    )
    fig.add_trace(ts_grid, row=2, col=2)

    # Vertical line for selected timestep
    vline_grid = go.Scatter(
        x=[0, 0],
        y=[df['grid_power'].min(), df['grid_power'].max()],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    )
    fig.add_trace(vline_grid, row=2, col=2)

    # Create slider steps
    steps = []
    for i in range(len(df)):
        step = dict(
            method="update",
            args=[
                {
                    "x": [
                        df['grid_power'],  # All points (scatter)
                        [df['grid_power'].iloc[i]],  # Highlighted point
                        None,  # Info table (no x)
                        df.index,  # Battery time series
                        [i, i],  # Battery vline
                        df.index,  # Grid time series
                        [i, i],  # Grid vline
                    ],
                    "y": [
                        df['battery_power'],  # All points (scatter)
                        [df['battery_power'].iloc[i]],  # Highlighted point
                        None,  # Info table (no y)
                        df['battery_power'],  # Battery time series
                        [df['battery_power'].min(), df['battery_power'].max()],  # Battery vline
                        df['grid_power'],  # Grid time series
                        [df['grid_power'].min(), df['grid_power'].max()],  # Grid vline
                    ],
                    "customdata": [
                        df[['timestamp', 'grid_power', 'battery_power', 'battery_soc', 'electricity_price']].values,
                        [df[['timestamp', 'grid_power', 'battery_power', 'battery_soc', 'electricity_price']].iloc[i].values],
                    ],
                    "hovertemplate": [
                        None,  # Keep scatter hovertemplate
                        '<b>SELECTED Step ' + str(i) + '</b><br>' +
                        'Time: %{customdata[0][0]}<br>' +
                        'Grid: %{customdata[0][1]:.3f} kW<br>' +
                        'Battery: %{customdata[0][2]:.3f} kW<br>' +
                        'SOC: %{customdata[0][3]:.3f}<br>' +
                        'Price: %{customdata[0][4]:.4f} €/kWh<br>' +
                        '<extra></extra>',
                    ],
                    "cells": [
                        None,  # Keep header
                        {
                            'values': [
                                ['Timestep', 'Time', 'Grid Power', 'Battery Power', 'SOC', 'Price'],
                                [
                                    str(i),
                                    df['timestamp'].iloc[i],
                                    f"{df['grid_power'].iloc[i]:.3f} kW",
                                    f"{df['battery_power'].iloc[i]:.3f} kW",
                                    f"{df['battery_soc'].iloc[i]:.3f}",
                                    f"{df['electricity_price'].iloc[i]:.4f} €/kWh"
                                ]
                            ]
                        }
                    ]
                }
            ],
            label=f"{i}"
        )
        steps.append(step)

    # Create slider
    sliders = [dict(
        active=0,
        yanchor="top",
        y=-0.05,
        xanchor="left",
        currentvalue=dict(
            prefix="Timestep: ",
            visible=True,
            xanchor="center",
            font=dict(size=14)
        ),
        pad=dict(b=10, t=10),
        len=0.9,
        x=0.05,
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        height=900,
        title=dict(
            text=f"<b>Interactive Battery Power (u) vs Grid Power - {controller_type}</b><br>" +
                 "<sub>Use slider below to navigate through timesteps | Hover over points for details</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        hovermode='closest',
        template='plotly_white'
    )

    # Update axes labels
    fig.update_xaxes(title_text="<b>Grid Power (kW)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Battery Power (kW)</b>", row=1, col=1)

    fig.update_xaxes(title_text="<b>Timestep</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Battery (kW)</b>", row=2, col=1)

    fig.update_xaxes(title_text="<b>Timestep</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>Grid (kW)</b>", row=2, col=2)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

    return fig

def main():
    """Main function."""
    # Get controller type from command line or config
    if len(sys.argv) > 1:
        controller_type = sys.argv[1]
        print(f"Using controller type from command line: {controller_type}")
    else:
        config = load_config()
        controller_type = config['controller']['type']
        print(f"Using controller type from config.yaml: {controller_type}")

    print(f"Loading results for controller: {controller_type}")

    # Load results
    df = load_results(controller_type)

    print(f"Loaded {len(df)} timesteps")
    print(f"Battery power range: [{df['battery_power'].min():.3f}, {df['battery_power'].max():.3f}] kW")
    print(f"Grid power range: [{df['grid_power'].min():.3f}, {df['grid_power'].max():.3f}] kW")

    # Create statistics
    print(f"\nStatistics:")
    print(f"  Battery charging (u > 0): {(df['battery_power'] > 0).sum()} steps ({(df['battery_power'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"  Battery discharging (u < 0): {(df['battery_power'] < 0).sum()} steps ({(df['battery_power'] < 0).sum()/len(df)*100:.1f}%)")
    print(f"  Battery idle (u = 0): {(df['battery_power'] == 0).sum()} steps ({(df['battery_power'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Grid import (grid > 0): {(df['grid_power'] > 0).sum()} steps ({(df['grid_power'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"  Grid export (grid < 0): {(df['grid_power'] < 0).sum()} steps ({(df['grid_power'] < 0).sum()/len(df)*100:.1f}%)")

    # Create interactive plot
    print("\nCreating interactive plot...")
    fig = create_interactive_plot(df, controller_type)

    # Save plot
    output_file = f"results/plots/u_vs_grid_{controller_type}.html"
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")
    print(f"Open the file in your browser to interact with the visualization.")

    # Also show in browser automatically
    print("\nOpening plot in browser...")
    fig.show()

if __name__ == "__main__":
    main()
