from typing import List

import altair as alt
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from gtmkit.utils.density import calculate_grid


def _infer_vega_type(series: pd.Series) -> str:
    """
    Infer a sensible Vega-Lite type code for Altair: 'Q', 'O', 'N', or 'T'.
    - Numbers (except bool) -> 'Q'
    - datetimes -> 'T'
    - categoricals -> 'O'
    - everything else -> 'N'
    """
    if is_bool_dtype(series):
        # Bool behaves better as nominal (two categories) than quantitative
        return "N"
    if is_datetime64_any_dtype(series):
        return "T"
    if is_numeric_dtype(series):
        return "Q"
    if is_categorical_dtype(series):
        # ordered categoricals are conceptually ordinal; nominal is also ok
        return "O" if getattr(series.dtype, "ordered", False) else "N"
    return "N"


def altair_points_chart(
    df,
    num_nodes: int,
    points_size: int = 12,
    coloring_scheme: str = "viridis",
    coloring_column: str = "color",
    legend=None,
    color_type: str = "auto",  # NEW: 'auto' | 'Q' | 'O' | 'N' | 'T'
):
    import altair as alt

    axis_len = int(num_nodes**0.5)

    # --- decide color encoding type ---
    if color_type == "auto":
        try:
            inferred = _infer_vega_type(df[coloring_column])
        except Exception:
            inferred = "Q"
        color_type = inferred

    legend_config = None if legend is None else legend

    color = alt.Color(
        f"{coloring_column}:{color_type}",
        legend=legend_config,
        scale=alt.Scale(scheme=coloring_scheme),
    )

    chart = (
        alt.Chart(df)
        .mark_point(opacity=0.9, filled=True, size=points_size)
        .encode(
            x=alt.X(
                "x:Q",
                title=None,
                axis=None,
                # centers at integers 1..N -> use 0.5..N+0.5
                scale=alt.Scale(domain=[1, axis_len + 1]),
            ),
            y=alt.Y(
                "y:Q",
                title=None,
                axis=None,
                scale=alt.Scale(domain=[1, axis_len + 1], reverse=True),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("x:Q"),
                alt.Tooltip("y:Q"),
                alt.Tooltip(f"{coloring_column}:{color_type}"),
            ],
        )
    )
    return chart


def altair_discrete_density_landscape(density_table, title=""):
    """
    It takes a density vector and returns an Altair-based visualisation of GTMap

    :param density: the density of the nodes
    :param node_threshold: minimal density, where values lower than specified means that the node is empty
    :return: A chart object
    """
    n_nodes = density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    chart = (
        alt.Chart(density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1)), reverse=True),
            ),
            color=alt.Color(
                "filtered_density:Q",
                title=None,
                scale=alt.Scale(scheme=alt.SchemeParams(name="greys")),
            ),
            tooltip=[
                alt.Tooltip("nodes", title="Node"),
                alt.Tooltip("density", title="Density"),
            ],
        )
    )

    return chart


def altair_discrete_class_landscape(
    class_density_table,
    colorset="lighttealblue",
    title="",
    use_density=False,
    first_class_prob_column_name="first_class_prob",
    second_class_prob_column_name="second_class_prob",
    first_class_density_column_name="first_class_density",
    second_class_density_column_name="second_class_density",
    first_class_label="Inactive",
    second_class_label="Active",
    reverse=False,
):
    n_nodes = class_density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    opacity = alt.Opacity()
    if use_density:
        opacity = alt.Opacity("density", title=None, legend=None)

    tooltip = [
        alt.Tooltip("nodes", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip(first_class_prob_column_name, title=first_class_label + " prob"),
        alt.Tooltip(second_class_prob_column_name, title=second_class_label + " prob"),
        alt.Tooltip(
            first_class_density_column_name, title=first_class_label + " density"
        ),
        alt.Tooltip(
            second_class_density_column_name, title=second_class_label + " density"
        ),
    ]

    chart = (
        alt.Chart(
            class_density_table,
            title=title,
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1)), reverse=True),
            ),
            color=alt.Color(
                f"{second_class_prob_column_name}:Q",
                title=None,
                scale=alt.Scale(
                    domain=[0, 1],
                    scheme=alt.SchemeParams(name=colorset),
                    reverse=reverse,
                ),
                legend=alt.Legend(),
            ),
            tooltip=tooltip,
            opacity=opacity,
        )
    )
    return chart


def altair_discrete_regression_landscape(
    reg_density_table,
    colorset="lighttealblue",
    use_density=False,
    scale_type="linear",
    regval_domain=None,
    reverse=False,
    title="",
):
    # --- minimal fix: derive grid size from table coords, not row count ---
    # works even if reg_density_table was filtered by node_threshold
    axis_len = int(max(reg_density_table["x"].max(), reg_density_table["y"].max()))

    opacity = alt.Opacity()
    if use_density:
        opacity = alt.Opacity("density", title=None, legend=None)

    tooltip = [
        alt.Tooltip("nodes", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip("filtered_reg_density", title="Regression density"),
    ]

    if regval_domain:
        color_scale = alt.Scale(
            domain=regval_domain,
            scheme=alt.SchemeParams(name=colorset),
            type=scale_type,
            reverse=reverse,
        )
    else:
        color_scale = alt.Scale(
            scheme=alt.SchemeParams(name=colorset),
            type=scale_type,
            reverse=reverse,
        )

    chart = (
        alt.Chart(reg_density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1)), reverse=True),
            ),
            color=alt.Color(
                "filtered_reg_density:Q",
                title=None,
                scale=color_scale,
                legend=alt.Legend(),
            ),
            tooltip=tooltip,
            opacity=opacity,
        )
    )
    return chart


def altair_discrete_query_landscape(
    class_density_table,
    colorset="viridis",
    title="",
    criteria_column="criteria_satisfied",
    reverse=True,
):
    """
    Generate landscape using Altair to visualize discrete criteria values over a 2D layout of nodes.
    Colors are applied only to cells with non-null criteria values.

    Parameters:
        class_density_table (pd.DataFrame): A DataFrame containing at least the following columns:
            - 'x': X grid coordinate (categorical or ordered)
            - 'y': Y grid coordinate (categorical or ordered)
            - 'nodes': Node IDs
            - 'density': A numeric value to include in tooltips
            - criteria_column (str): A column with discrete criteria values (can include NaN)

        colorset (str): Name of the color scheme to use (e.g., 'viridis', 'category10').
        title (str): Optional title to display above the chart.
        criteria_column (str): Column used for determining coloring of the cells.
        reverse (bool): Whether to reverse the color scale (default is True).

    Returns:
        alt.Chart: An Altair chart object representing the discrete query landscape.
    """
    n_nodes = class_density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    non_null_values = class_density_table[criteria_column].dropna().unique().tolist()
    non_null_values = sorted(non_null_values)

    tooltip = [
        alt.Tooltip("nodes:O", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip(f"{criteria_column}:N", title="Criteria Satisfied"),
    ]

    chart = (
        alt.Chart(class_density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            color=alt.condition(
                f"datum['{criteria_column}'] !== null",
                alt.Color(
                    f"{criteria_column}:N",
                    title="Criteria Satisfied",
                    scale=alt.Scale(
                        scheme=colorset, domain=non_null_values, reverse=reverse
                    ),
                    legend=alt.Legend(),
                ),
                alt.value("white"),
            ),
            tooltip=tooltip,
        )
    )
    return chart
