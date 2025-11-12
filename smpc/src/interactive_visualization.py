"""
Interactive 3D Visualization for SDP Interpolation

This module provides interactive 3D visualizations to understand the bilinear
interpolation happening in the SDP controller. It shows the 27 nearest grid
points and how they are used for interpolation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd


class InterpolationVisualizer:
    """
    Visualizer for SDP interpolation in 3D.

    Shows the 27 nearest grid points (3x3x3 cube) around the current state
    and displays the decision U and cost J on the Z axis.
    """

    def __init__(self, output_dir: str = "results/interactive_plots"):
        """
        Args:
            output_dir: Directory to save HTML plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for visualization data at each timestep
        self.timestep_data = []

    def add_timestep_data(self,
                         timestamp: str,
                         x_current: float,
                         R_current: float,
                         k: int,
                         X_grid: np.ndarray,
                         R_grid: np.ndarray,
                         policy: np.ndarray,
                         value_function: np.ndarray,
                         u_star: float,
                         J_star: float):
        """
        Store data for a timestep for later visualization.

        Args:
            timestamp: Current timestamp as string
            x_current: Current SOC (state of charge)
            R_current: Current residual (P_pv - P_load)
            k: Current timestep index in horizon
            X_grid: SOC grid points
            R_grid: Residual grid points for this k
            policy: Policy matrix [n_x, n_R] with optimal actions
            value_function: Value function matrix [n_x, n_R] with costs
            u_star: Interpolated optimal action
            J_star: Interpolated optimal cost
        """
        # Convert to numpy arrays if needed (for Numba compatibility)
        X_grid = np.asarray(X_grid)
        R_grid = np.asarray(R_grid)
        policy = np.asarray(policy)
        value_function = np.asarray(value_function)

        # Validation and debugging
        if len(self.timestep_data) == 0:  # First data point
            print(f"\n[Visualization Debug] First timestep data:")
            print(f"  X_grid shape: {X_grid.shape}, range: [{X_grid.min():.3f}, {X_grid.max():.3f}]")
            print(f"  R_grid shape: {R_grid.shape}, range: [{R_grid.min():.3f}, {R_grid.max():.3f}]")
            print(f"  policy shape: {policy.shape}, range: [{policy.min():.3f}, {policy.max():.3f}]")
            print(f"  value_function shape: {value_function.shape}, range: [{value_function.min():.3f}, {value_function.max():.3f}]")
            print(f"  u_star: {u_star:.3f}, J_star: {J_star:.3f}")
            print(f"  Has NaN in policy: {np.isnan(policy).any()}")
            print(f"  Has Inf in policy: {np.isinf(policy).any()}")

        data = {
            'timestamp': timestamp,
            'x_current': x_current,
            'R_current': R_current,
            'k': k,
            'X_grid': X_grid.copy(),
            'R_grid': R_grid.copy(),
            'policy': policy.copy(),
            'value_function': value_function.copy(),
            'u_star': u_star,
            'J_star': J_star
        }
        self.timestep_data.append(data)

    def get_nearest_grid_points(self,
                                x_current: float,
                                R_current: float,
                                X_grid: np.ndarray,
                                R_grid: np.ndarray,
                                n_neighbors: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest grid points around current state.

        Returns n_neighbors x n_neighbors points centered around (x_current, R_current).

        Args:
            x_current: Current SOC
            R_current: Current residual
            X_grid: SOC grid points
            R_grid: Residual grid points
            n_neighbors: Number of neighbors in each dimension (default 3 for 3x3)

        Returns:
            (X_nearest, R_nearest, indices_x, indices_R)
        """
        # Find nearest index in X_grid
        i_x = np.searchsorted(X_grid, x_current)
        i_x = np.clip(i_x, 1, len(X_grid) - 1)

        # Find nearest index in R_grid
        i_R = np.searchsorted(R_grid, R_current)
        i_R = np.clip(i_R, 1, len(R_grid) - 1)

        # Get n_neighbors points centered around the current state
        half = n_neighbors // 2

        # X indices
        i_x_start = max(0, i_x - half)
        i_x_end = min(len(X_grid), i_x + half + 1)
        indices_x = np.arange(i_x_start, i_x_end)

        # R indices
        i_R_start = max(0, i_R - half)
        i_R_end = min(len(R_grid), i_R + half + 1)
        indices_R = np.arange(i_R_start, i_R_end)

        X_nearest = X_grid[indices_x]
        R_nearest = R_grid[indices_R]

        return X_nearest, R_nearest, indices_x, indices_R

    def get_bilinear_interpolation_corners(self,
                                          x_current: float,
                                          R_current: float,
                                          X_grid: np.ndarray,
                                          R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the 4 corner points used in bilinear interpolation.

        Returns:
            (X_corners, R_corners, indices_x, indices_R) where each is array of length 4
        """
        # Find nearest index in X_grid
        i_x = np.searchsorted(X_grid, x_current)
        i_x = np.clip(i_x, 1, len(X_grid) - 1)

        # Find nearest index in R_grid
        i_R = np.searchsorted(R_grid, R_current)
        i_R = np.clip(i_R, 1, len(R_grid) - 1)

        # Get the 4 corners: (low_x, low_R), (low_x, high_R), (high_x, low_R), (high_x, high_R)
        indices_x = np.array([i_x - 1, i_x - 1, i_x, i_x])
        indices_R = np.array([i_R - 1, i_R, i_R - 1, i_R])

        X_corners = X_grid[indices_x]
        R_corners = R_grid[indices_R]

        return X_corners, R_corners, indices_x, indices_R

    def create_3d_plot_decision(self,
                               data_idx: int,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D scatter plot showing decision U with nearest grid points.

        Args:
            data_idx: Index in timestep_data to visualize
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure
        """
        if data_idx >= len(self.timestep_data):
            raise ValueError(f"data_idx {data_idx} out of range (only {len(self.timestep_data)} timesteps)")

        data = self.timestep_data[data_idx]

        # Use ALL grid points (not just nearest)
        X_grid = data['X_grid']
        R_grid = data['R_grid']

        # Create meshgrid for ALL points
        X_mesh, R_mesh = np.meshgrid(X_grid, R_grid, indexing='ij')

        # Extract policy values for ALL points
        U_all = data['policy']

        # Flatten for scatter plot
        X_flat = X_mesh.flatten()
        R_flat = R_mesh.flatten()
        U_flat = U_all.flatten()

        # Debug info
        print(f"\n[Plot Debug] Decision U at timestep {data_idx}:")
        print(f"  Total grid points: {len(X_flat)} ({len(X_grid)} SOC × {len(R_grid)} R)")
        print(f"  X range: [{X_flat.min():.3f}, {X_flat.max():.3f}]")
        print(f"  R range: [{R_flat.min():.3f}, {R_flat.max():.3f}]")
        print(f"  U range: [{U_flat.min():.3f}, {U_flat.max():.3f}]")
        print(f"  U has NaN: {np.isnan(U_flat).any()}, Inf: {np.isinf(U_flat).any()}")

        # Filter out invalid values (NaN, Inf)
        valid_mask = np.isfinite(U_flat)
        if not valid_mask.all():
            print(f"  Warning: Found {(~valid_mask).sum()} invalid U values, filtering...")
            X_flat = X_flat[valid_mask]
            R_flat = R_flat[valid_mask]
            U_flat = U_flat[valid_mask]

        # Clamp unrealistic values (battery actions should be within -10 to 10 kW typically)
        U_reasonable = np.abs(U_flat) < 100  # More generous limit for edge cases
        if not U_reasonable.all():
            print(f"  Warning: Found {(~U_reasonable).sum()} unreasonable U values (|U| > 100), clamping...")
            U_flat = np.clip(U_flat, -100, 100)

        # Get the 4 corner points used in bilinear interpolation
        X_corners, R_corners, idx_x_corners, idx_R_corners = self.get_bilinear_interpolation_corners(
            data['x_current'],
            data['R_current'],
            data['X_grid'],
            data['R_grid']
        )
        U_corners = data['policy'][idx_x_corners, idx_R_corners]
        U_corners = np.clip(U_corners, -100, 100)

        # Create figure
        fig = go.Figure()

        # Add ALL grid points (very small, semi-transparent)
        fig.add_trace(go.Scatter3d(
            x=X_flat,
            y=R_flat,
            z=U_flat,
            mode='markers',
            marker=dict(
                size=2,  # Smaller for many points
                color=U_flat,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="U (kW)", x=1.1),
                opacity=0.4,
                line=dict(width=0)
            ),
            name='All Grid Points',
            text=[f'SOC: {x:.3f}<br>R: {r:.3f}<br>U: {u:.3f}'
                  for x, r, u in zip(X_flat, R_flat, U_flat)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Add the 4 INTERPOLATION CORNERS (highlighted)
        fig.add_trace(go.Scatter3d(
            x=X_corners,
            y=R_corners,
            z=U_corners,
            mode='markers+text',
            marker=dict(
                size=12,
                color='orange',
                symbol='square',
                line=dict(width=2, color='darkorange')
            ),
            text=[f'U={u:.2f}' for u in U_corners],
            textposition='top center',
            textfont=dict(size=10, color='orange'),
            name='Interpolation Corners',
            hovertemplate='Corner Point<br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>U: %{z:.3f}<extra></extra>'
        ))

        # Add lines from corners to current state
        for i in range(4):
            fig.add_trace(go.Scatter3d(
                x=[X_corners[i], data['x_current']],
                y=[R_corners[i], data['R_current']],
                z=[U_corners[i], data['u_star']],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add current state point (INTERPOLATED, larger and prominent)
        fig.add_trace(go.Scatter3d(
            x=[data['x_current']],
            y=[data['R_current']],
            z=[data['u_star']],
            mode='markers+text',
            marker=dict(
                size=18,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='darkred')
            ),
            text=[f"U*={data['u_star']:.2f}"],
            textposition='top center',
            textfont=dict(size=12, color='red', family='Arial Black'),
            name='Interpolated Value',
            hovertemplate='<b>INTERPOLATED</b><br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>U*: %{z:.3f}<extra></extra>'
        ))

        # Surface removed per user request

        # Update layout
        fig.update_layout(
            title=f"Decision U - Interpolation Visualization<br>Timestamp: {data['timestamp']}, k={data['k']}",
            scene=dict(
                xaxis_title='SOC (State of Charge)',
                yaxis_title='R (Residual P_pv - P_load) [kW]',
                zaxis_title='U (Battery Action) [kW]',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=800,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved decision plot to {save_path}")

        return fig

    def create_3d_plot_cost(self,
                           data_idx: int,
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D scatter plot showing cost J with nearest grid points.

        Args:
            data_idx: Index in timestep_data to visualize
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure
        """
        if data_idx >= len(self.timestep_data):
            raise ValueError(f"data_idx {data_idx} out of range (only {len(self.timestep_data)} timesteps)")

        data = self.timestep_data[data_idx]

        # Use ALL grid points (not just nearest)
        X_grid = data['X_grid']
        R_grid = data['R_grid']

        # Create meshgrid for ALL points
        X_mesh, R_mesh = np.meshgrid(X_grid, R_grid, indexing='ij')

        # Extract value function for ALL points
        J_all = data['value_function']

        # Flatten for scatter plot
        X_flat = X_mesh.flatten()
        R_flat = R_mesh.flatten()
        J_flat = J_all.flatten()

        # Debug info
        print(f"\n[Plot Debug] Cost J at timestep {data_idx}:")
        print(f"  Total grid points: {len(X_flat)} ({len(X_grid)} SOC × {len(R_grid)} R)")
        print(f"  X range: [{X_flat.min():.3f}, {X_flat.max():.3f}]")
        print(f"  R range: [{R_flat.min():.3f}, {R_flat.max():.3f}]")
        print(f"  J range: [{J_flat.min():.3f}, {J_flat.max():.3f}]")
        print(f"  J has NaN: {np.isnan(J_flat).any()}, Inf: {np.isinf(J_flat).any()}")

        # Filter out invalid values (NaN, Inf)
        valid_mask = np.isfinite(J_flat)
        if not valid_mask.all():
            print(f"  Warning: Found {(~valid_mask).sum()} invalid J values, filtering...")
            X_flat = X_flat[valid_mask]
            R_flat = R_flat[valid_mask]
            J_flat = J_flat[valid_mask]

        # Clamp unrealistic values (costs should be reasonable, say < 10000 €)
        J_reasonable = J_flat < 10000
        if not J_reasonable.all():
            print(f"  Warning: Found {(~J_reasonable).sum()} unreasonable J values (J > 10000), clamping...")
            J_flat = np.clip(J_flat, 0, 10000)

        # Get the 4 corner points used in bilinear interpolation
        X_corners, R_corners, idx_x_corners, idx_R_corners = self.get_bilinear_interpolation_corners(
            data['x_current'],
            data['R_current'],
            data['X_grid'],
            data['R_grid']
        )
        J_corners = data['value_function'][idx_x_corners, idx_R_corners]
        J_corners = np.clip(J_corners, -10000, 10000)

        # Create figure
        fig = go.Figure()

        # Add ALL grid points (very small, semi-transparent)
        fig.add_trace(go.Scatter3d(
            x=X_flat,
            y=R_flat,
            z=J_flat,
            mode='markers',
            marker=dict(
                size=2,  # Smaller for many points
                color=J_flat,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="J (Cost) [€]", x=1.1),
                opacity=0.4,
                line=dict(width=0)
            ),
            name='All Grid Points',
            text=[f'SOC: {x:.3f}<br>R: {r:.3f}<br>J: {j:.4f}'
                  for x, r, j in zip(X_flat, R_flat, J_flat)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Add the 4 INTERPOLATION CORNERS (highlighted)
        fig.add_trace(go.Scatter3d(
            x=X_corners,
            y=R_corners,
            z=J_corners,
            mode='markers+text',
            marker=dict(
                size=12,
                color='cyan',
                symbol='square',
                line=dict(width=2, color='darkcyan')
            ),
            text=[f'J={j:.3f}' for j in J_corners],
            textposition='top center',
            textfont=dict(size=10, color='cyan'),
            name='Interpolation Corners',
            hovertemplate='Corner Point<br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>J: %{z:.4f}<extra></extra>'
        ))

        # Add lines from corners to current state
        for i in range(4):
            fig.add_trace(go.Scatter3d(
                x=[X_corners[i], data['x_current']],
                y=[R_corners[i], data['R_current']],
                z=[J_corners[i], data['J_star']],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add current state point (INTERPOLATED, larger and prominent)
        fig.add_trace(go.Scatter3d(
            x=[data['x_current']],
            y=[data['R_current']],
            z=[data['J_star']],
            mode='markers+text',
            marker=dict(
                size=18,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='darkred')
            ),
            text=[f"J*={data['J_star']:.3f}"],
            textposition='top center',
            textfont=dict(size=12, color='red', family='Arial Black'),
            name='Interpolated Value',
            hovertemplate='<b>INTERPOLATED</b><br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>J*: %{z:.4f}<extra></extra>'
        ))

        # Surface removed per user request

        # Update layout
        fig.update_layout(
            title=f"Cost J - Interpolation Visualization<br>Timestamp: {data['timestamp']}, k={data['k']}",
            scene=dict(
                xaxis_title='SOC (State of Charge)',
                yaxis_title='R (Residual P_pv - P_load) [kW]',
                zaxis_title='J (Cost-to-go) [€]',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=800,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved cost plot to {save_path}")

        return fig

    def create_combined_plot_with_slider(self,
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create a combined plot with slider to navigate through timesteps.

        Shows both U and J side by side with a slider to move through time.

        Args:
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure with slider
        """
        if len(self.timestep_data) == 0:
            raise ValueError("No timestep data available. Call add_timestep_data first.")

        # Create subplots
        from plotly.subplots import make_subplots

        # We'll create frames for animation
        frames = []

        # Get first timestep for initial plot
        data = self.timestep_data[0]

        # Create initial figure with both decision and cost
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Decision U (Battery Action)', 'Cost J (Cost-to-go)'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.1
        )

        # Function to create traces for a given timestep
        def create_traces(data):
            """Helper to create traces for both plots"""
            # Use ALL grid points
            X_grid = data['X_grid']
            R_grid = data['R_grid']

            # Create meshgrid for ALL points
            X_mesh, R_mesh = np.meshgrid(X_grid, R_grid, indexing='ij')

            # Extract values for ALL points
            U_all = data['policy']
            J_all = data['value_function']

            # Flatten
            X_flat = X_mesh.flatten()
            R_flat = R_mesh.flatten()
            U_flat = U_all.flatten()
            J_flat = J_all.flatten()

            # Filter invalid values and clamp to reasonable ranges
            valid_U_mask = np.isfinite(U_flat)
            valid_J_mask = np.isfinite(J_flat)

            if not valid_U_mask.all():
                U_flat = np.where(valid_U_mask, U_flat, 0.0)
                U_all = np.where(np.isfinite(U_all), U_all, 0.0)

            if not valid_J_mask.all():
                J_flat = np.where(valid_J_mask, J_flat, 0.0)
                J_all = np.where(np.isfinite(J_all), J_all, 0.0)

            # Clamp unrealistic values (same as in individual plots)
            U_flat = np.clip(U_flat, -100, 100)
            U_all = np.clip(U_all, -100, 100)
            J_flat = np.clip(J_flat, -10000, 10000)
            J_all = np.clip(J_all, -10000, 10000)

            traces = []

            # Decision U traces (col 1) - ALL points
            traces.append(go.Scatter3d(
                x=X_flat, y=R_flat, z=U_flat,
                mode='markers',
                marker=dict(size=2, color=U_flat, colorscale='Viridis',
                           showscale=False, opacity=0.4, line=dict(width=0)),
                name='Grid U',
                showlegend=False,
                hovertemplate='SOC: %{x:.3f}<br>R: %{y:.3f}<br>U: %{z:.3f}<extra></extra>'
            ))

            traces.append(go.Scatter3d(
                x=[data['x_current']], y=[data['R_current']], z=[data['u_star']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond',
                           line=dict(width=2, color='darkred')),
                name='Current U',
                showlegend=False,
                hovertemplate='Current<br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>U*: %{z:.3f}<extra></extra>'
            ))

            # Surface U removed per user request

            # Cost J traces (col 2) - ALL points
            traces.append(go.Scatter3d(
                x=X_flat, y=R_flat, z=J_flat,
                mode='markers',
                marker=dict(size=2, color=J_flat, colorscale='Plasma',
                           showscale=False, opacity=0.4, line=dict(width=0)),
                name='Grid J',
                showlegend=False,
                hovertemplate='SOC: %{x:.3f}<br>R: %{y:.3f}<br>J: %{z:.4f}<extra></extra>'
            ))

            traces.append(go.Scatter3d(
                x=[data['x_current']], y=[data['R_current']], z=[data['J_star']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond',
                           line=dict(width=2, color='darkred')),
                name='Current J',
                showlegend=False,
                hovertemplate='Current<br>SOC: %{x:.3f}<br>R: %{y:.3f}<br>J*: %{z:.4f}<extra></extra>'
            ))

            # Surface J removed per user request

            return traces

        # Add initial traces
        traces = create_traces(data)
        for i, trace in enumerate(traces):
            if i < 3:  # U traces
                fig.add_trace(trace, row=1, col=1)
            else:  # J traces
                fig.add_trace(trace, row=1, col=2)

        # Create frames for slider
        for idx, data in enumerate(self.timestep_data):
            traces = create_traces(data)

            frame = go.Frame(
                data=traces,
                name=str(idx),
                layout=go.Layout(
                    title_text=f"Interpolation Visualization - {data['timestamp']} (k={data['k']})"
                )
            )
            frames.append(frame)

        fig.frames = frames

        # Add slider
        sliders = [dict(
            active=0,
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1,
            currentvalue=dict(
                prefix="Timestep: ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=50),
            len=0.8,
            steps=[dict(
                args=[[f.name], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0)
                )],
                label=self.timestep_data[k]['timestamp'],
                method="animate"
            ) for k, f in enumerate(frames)]
        )]

        # Update layout
        fig.update_layout(
            title=f"SDP Interpolation - Decision U and Cost J<br>First timestep: {data['timestamp']}",
            sliders=sliders,
            width=1600,
            height=700,
            showlegend=False
        )

        # Calculate global Z-axis ranges across all timesteps for consistent scaling
        all_U_values = []
        all_J_values = []
        for ts_data in self.timestep_data:
            policy_data = np.asarray(ts_data['policy'])
            value_data = np.asarray(ts_data['value_function'])
            # Clamp before collecting ranges
            policy_data = np.clip(policy_data[np.isfinite(policy_data)], -100, 100)
            value_data = np.clip(value_data[np.isfinite(value_data)], -10000, 10000)
            all_U_values.extend(policy_data.flatten())
            all_J_values.extend(value_data.flatten())

        # Calculate reasonable ranges with some padding
        U_min, U_max = np.percentile(all_U_values, [1, 99])  # Use 1st/99th percentile to ignore outliers
        J_min, J_max = np.percentile(all_J_values, [1, 99])
        U_padding = (U_max - U_min) * 0.1
        J_padding = (J_max - J_min) * 0.1

        print(f"\n[Combined Plot] Fixed Z-axis ranges:")
        print(f"  U (Decision): [{U_min - U_padding:.3f}, {U_max + U_padding:.3f}] kW")
        print(f"  J (Cost): [{J_min - J_padding:.3f}, {J_max + J_padding:.3f}] €")

        # Update 3D scene settings with fixed Z ranges
        fig.update_scenes(
            xaxis_title='SOC',
            yaxis_title='R [kW]',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )

        # Separate Z-axis titles and ranges for each subplot
        fig.update_scenes(
            zaxis_title='U [kW]',
            zaxis=dict(range=[U_min - U_padding, U_max + U_padding]),
            row=1, col=1
        )
        fig.update_scenes(
            zaxis_title='J [€]',
            zaxis=dict(range=[J_min - J_padding, J_max + J_padding]),
            row=1, col=2
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved combined plot with slider to {save_path}")

        return fig

    def save_all_plots(self, prefix: str = "interpolation"):
        """
        Save all available plots to the output directory.

        Args:
            prefix: Prefix for filenames
        """
        if len(self.timestep_data) == 0:
            print("No timestep data to visualize.")
            return

        # Save combined plot with slider
        combined_path = self.output_dir / f"{prefix}_combined_slider.html"
        self.create_combined_plot_with_slider(save_path=str(combined_path))

        # Optionally save individual timesteps (just first, middle, last)
        indices = [0, len(self.timestep_data) // 2, len(self.timestep_data) - 1]

        for idx in indices:
            if idx < len(self.timestep_data):
                ts = self.timestep_data[idx]['timestamp'].replace(':', '-').replace(' ', '_')

                u_path = self.output_dir / f"{prefix}_decision_U_{ts}.html"
                self.create_3d_plot_decision(idx, save_path=str(u_path))

                j_path = self.output_dir / f"{prefix}_cost_J_{ts}.html"
                self.create_3d_plot_cost(idx, save_path=str(j_path))

        print(f"\nAll plots saved to {self.output_dir}")