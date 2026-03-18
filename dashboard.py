"""
Epidemiological Pulse – Interactive Dashboard
===============================================
Plotly Dash web application providing:
  - Interactive choropleth/scatter map of disease hotspots
  - Time-series charts with Prophet forecast bands
  - Risk level indicators per region
  - Filters: ZIP code, date range, risk level
  - Auto-refresh support
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color Scheme
# ---------------------------------------------------------------------------
COLORS = {
    "background": "#0f1117",
    "card": "#1a1d27",
    "card_border": "#2d3047",
    "text_primary": "#e8eaf6",
    "text_secondary": "#9fa8da",
    "accent_blue": "#5c6bc0",
    "accent_cyan": "#26c6da",
    "risk_low": "#2ecc71",
    "risk_medium": "#f39c12",
    "risk_high": "#e74c3c",
    "chart_bg": "#131620",
}

RISK_COLOR_MAP = {
    "Low": COLORS["risk_low"],
    "Medium": COLORS["risk_medium"],
    "High": COLORS["risk_high"],
}


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class DashboardDataLoader:
    """Loads and caches data for the dashboard."""

    def __init__(
        self,
        master_csv_path: str,
        geojson_path: str,
        forecast_dir: str,
        config: Dict,
    ) -> None:
        self.master_csv_path = master_csv_path
        self.geojson_path = geojson_path
        self.forecast_dir = forecast_dir
        self.config = config
        self._master_df: Optional[pd.DataFrame] = None
        self._geojson: Optional[Dict] = None
        self._forecasts: Dict[str, pd.DataFrame] = {}

    def load_master(self) -> pd.DataFrame:
        """Load and cache master dataset."""
        if self._master_df is None:
            if not os.path.exists(self.master_csv_path):
                raise FileNotFoundError(f"Master dataset not found: {self.master_csv_path}")
            self._master_df = pd.read_csv(self.master_csv_path, parse_dates=["date"])
            logger.info(f"Loaded master dataset: {len(self._master_df):,} rows")
        return self._master_df

    def load_geojson(self) -> Optional[Dict]:
        """Load and cache GeoJSON."""
        if self._geojson is None:
            if os.path.exists(self.geojson_path):
                with open(self.geojson_path) as f:
                    self._geojson = json.load(f)
                logger.info(f"Loaded GeoJSON: {len(self._geojson.get('features', []))} features")
            else:
                logger.warning(f"GeoJSON not found: {self.geojson_path}")
        return self._geojson

    def get_region_summary(self, window_days: int = 30) -> pd.DataFrame:
        """Get recent regional summary for map display."""
        df = self.load_master()
        cutoff = df["date"].max() - pd.Timedelta(days=window_days)
        recent = df[df["date"] >= cutoff]

        agg = (
            recent.groupby("zipcode")
            .agg(
                {
                    "er_visits": "mean",
                    "pharmacy_sales": "mean",
                    "negative_sentiment": "mean",
                    "aqi": "mean",
                    "disease_risk_score": "mean",
                    "lat": "first",
                    "lon": "first",
                    "city": "first",
                    "state": "first",
                }
            )
            .reset_index()
        )

        # Add risk levels from GeoJSON if available
        geojson = self.load_geojson()
        if geojson:
            risk_map = {
                f["properties"]["zipcode"]: f["properties"]["risk_level"]
                for f in geojson.get("features", [])
            }
            agg["risk_level"] = agg["zipcode"].astype(str).map(risk_map).fillna("Low")
        else:
            agg["risk_level"] = agg["disease_risk_score"].apply(
                lambda s: "High" if s > 0.66 else ("Medium" if s > 0.33 else "Low")
            )

        return agg

    def get_region_timeseries(
        self,
        zipcode: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get time series data for a specific region."""
        df = self.load_master()
        region_df = df[df["zipcode"].astype(str) == str(zipcode)].copy()

        if start_date:
            region_df = region_df[region_df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            region_df = region_df[region_df["date"] <= pd.to_datetime(end_date)]

        return region_df.sort_values("date")

    def load_prophet_forecast(self, zipcode: str) -> Optional[pd.DataFrame]:
        """Load a Prophet forecast CSV for a region."""
        cache_key = f"prophet_{zipcode}"
        if cache_key not in self._forecasts:
            fname = os.path.join(self.forecast_dir, f"forecast_{zipcode}.csv")
            if os.path.exists(fname):
                self._forecasts[cache_key] = pd.read_csv(fname, parse_dates=["ds"])
            else:
                return None
        return self._forecasts.get(cache_key)

    def get_all_zipcodes(self) -> List[str]:
        """Return sorted list of all ZIP codes."""
        df = self.load_master()
        return sorted(df["zipcode"].astype(str).unique().tolist())

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Return overall date range of the dataset."""
        df = self.load_master()
        return df["date"].min().to_pydatetime(), df["date"].max().to_pydatetime()


# ---------------------------------------------------------------------------
# Chart Builders
# ---------------------------------------------------------------------------

class ChartBuilder:
    """Builds Plotly figure objects for the dashboard."""

    @staticmethod
    def hotspot_map(region_summary: pd.DataFrame) -> go.Figure:
        """
        Build an interactive scatter map of disease hotspots.

        Args:
            region_summary: Region-level summary with lat, lon, risk_level.

        Returns:
            Plotly figure.
        """
        if region_summary.empty or "lat" not in region_summary.columns:
            return go.Figure().update_layout(
                paper_bgcolor=COLORS["card"],
                plot_bgcolor=COLORS["chart_bg"],
                title="No geospatial data available",
            )

        fig = px.scatter_mapbox(
            region_summary,
            lat="lat",
            lon="lon",
            color="risk_level",
            size="disease_risk_score",
            size_max=40,
            color_discrete_map=RISK_COLOR_MAP,
            hover_name="city",
            hover_data={
                "zipcode": True,
                "state": True,
                "er_visits": ":.1f",
                "pharmacy_sales": ":.1f",
                "aqi": ":.1f",
                "disease_risk_score": ":.3f",
                "risk_level": True,
                "lat": False,
                "lon": False,
            },
            zoom=3.5,
            center={"lat": 38.0, "lon": -97.0},
            mapbox_style="open-street-map",
            title="Disease Hotspot Map (Last 30 Days)",
        )

        fig.update_layout(
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["chart_bg"],
            font_color=COLORS["text_primary"],
            title_font_size=16,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(
                bgcolor=COLORS["card"],
                bordercolor=COLORS["card_border"],
                font_color=COLORS["text_primary"],
            ),
            height=480,
        )
        return fig

    @staticmethod
    def er_timeseries(
        ts_df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        zipcode: str = "",
    ) -> go.Figure:
        """
        Build ER visits time-series chart with optional forecast band.

        Args:
            ts_df: Historical time series DataFrame.
            forecast_df: Prophet forecast DataFrame (ds, yhat, yhat_lower, yhat_upper).
            zipcode: ZIP code label.

        Returns:
            Plotly figure.
        """
        fig = go.Figure()

        # Actual values
        fig.add_trace(
            go.Scatter(
                x=ts_df["date"],
                y=ts_df["er_visits"],
                mode="lines",
                name="Actual ER Visits",
                line=dict(color=COLORS["accent_cyan"], width=2),
            )
        )

        # Anomaly markers
        anomaly_col = "er_visits_anomaly"
        if anomaly_col in ts_df.columns:
            anomalies = ts_df[ts_df[anomaly_col] == 1]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["date"],
                        y=anomalies["er_visits"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            color=COLORS["risk_high"],
                            size=10,
                            symbol="x",
                        ),
                    )
                )

        # Forecast
        if forecast_df is not None and not forecast_df.empty:
            # Future only
            last_actual = ts_df["date"].max()
            future_fc = forecast_df[forecast_df["ds"] > last_actual]

            if not future_fc.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([pd.Series([last_actual]), future_fc["ds"]]),
                        y=pd.concat(
                            [
                                ts_df[ts_df["date"] == last_actual]["er_visits"],
                                future_fc["yhat"],
                            ]
                        ),
                        mode="lines",
                        name="Forecast",
                        line=dict(color=COLORS["risk_medium"], width=2, dash="dash"),
                    )
                )

                # Confidence band
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
                        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(243, 156, 18, 0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=True,
                        name="Forecast CI",
                    )
                )

        fig.update_layout(
            title=f"ER Visits – ZIP {zipcode}",
            xaxis_title="Date",
            yaxis_title="ER Visits per Day",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["chart_bg"],
            font_color=COLORS["text_primary"],
            xaxis=dict(gridcolor=COLORS["card_border"]),
            yaxis=dict(gridcolor=COLORS["card_border"]),
            legend=dict(
                bgcolor=COLORS["card"],
                bordercolor=COLORS["card_border"],
            ),
            height=300,
            margin=dict(l=60, r=20, t=50, b=50),
        )
        return fig

    @staticmethod
    def signal_timeseries(ts_df: pd.DataFrame, zipcode: str = "") -> go.Figure:
        """
        Multi-signal time series chart (pharmacy, sentiment, AQI).

        Args:
            ts_df: Historical time series DataFrame.
            zipcode: ZIP code label.

        Returns:
            Plotly figure.
        """
        fig = go.Figure()

        signal_map = {
            "pharmacy_sales": ("Pharmacy Sales", COLORS["accent_blue"], "y1"),
            "negative_sentiment": ("Negative Sentiment", "#ef5350", "y2"),
            "aqi": ("AQI", "#ab47bc", "y3"),
        }

        for col, (name, color, _) in signal_map.items():
            if col in ts_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["date"],
                        y=ts_df[col],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.5),
                        opacity=0.85,
                    )
                )

        fig.update_layout(
            title=f"Health Signals – ZIP {zipcode}",
            xaxis_title="Date",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["chart_bg"],
            font_color=COLORS["text_primary"],
            xaxis=dict(gridcolor=COLORS["card_border"]),
            yaxis=dict(gridcolor=COLORS["card_border"]),
            legend=dict(
                bgcolor=COLORS["card"],
                bordercolor=COLORS["card_border"],
                orientation="h",
                y=-0.25,
            ),
            height=260,
            margin=dict(l=60, r=20, t=50, b=60),
        )
        return fig

    @staticmethod
    def risk_distribution_bar(region_summary: pd.DataFrame) -> go.Figure:
        """Bar chart showing risk level distribution across all regions."""
        counts = region_summary["risk_level"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        )
        fig = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[RISK_COLOR_MAP.get(r, "#95a5a6") for r in counts.index],
                text=counts.values,
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Risk Distribution Across Regions",
            xaxis_title="Risk Level",
            yaxis_title="Number of Regions",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["chart_bg"],
            font_color=COLORS["text_primary"],
            xaxis=dict(gridcolor=COLORS["card_border"]),
            yaxis=dict(gridcolor=COLORS["card_border"]),
            showlegend=False,
            height=280,
            margin=dict(l=60, r=20, t=50, b=50),
        )
        return fig

    @staticmethod
    def risk_score_heatmap(df: pd.DataFrame, n_regions: int = 15) -> go.Figure:
        """Heatmap of disease risk score over time per region."""
        # Pivot: zipcode × date
        recent = df.copy()
        recent["date"] = pd.to_datetime(recent["date"])
        cutoff = recent["date"].max() - pd.Timedelta(days=60)
        recent = recent[recent["date"] >= cutoff]

        if recent.empty:
            return go.Figure()

        pivot = (
            recent.pivot_table(
                index="zipcode",
                columns="date",
                values="disease_risk_score",
                aggfunc="mean",
            )
            .fillna(0)
        )

        top_zips = (
            recent.groupby("zipcode")["disease_risk_score"]
            .mean()
            .nlargest(n_regions)
            .index
        )
        pivot = pivot.loc[pivot.index.isin(top_zips)]
        pivot = pivot.reindex(sorted(pivot.index))

        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[str(d.date()) for d in pivot.columns],
                y=[str(z) for z in pivot.index],
                colorscale=[
                    [0.0, COLORS["risk_low"]],
                    [0.5, COLORS["risk_medium"]],
                    [1.0, COLORS["risk_high"]],
                ],
                colorbar=dict(
                    title="Risk Score",
                    tickfont=dict(color=COLORS["text_primary"]),
                    titlefont=dict(color=COLORS["text_primary"]),
                ),
                hoverongaps=False,
                zmin=0,
                zmax=1,
            )
        )
        fig.update_layout(
            title="Disease Risk Score Heatmap (Top 15 Regions – Last 60 Days)",
            xaxis_title="Date",
            yaxis_title="ZIP Code",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["chart_bg"],
            font_color=COLORS["text_primary"],
            height=380,
            margin=dict(l=80, r=20, t=50, b=60),
        )
        return fig


# ---------------------------------------------------------------------------
# Dashboard App Builder
# ---------------------------------------------------------------------------

def create_app(
    master_csv_path: str,
    geojson_path: str,
    forecast_dir: str,
    config_path: str = "config/params.yaml",
) -> Dash:
    """
    Create and configure the Dash application.

    Args:
        master_csv_path: Path to the master CSV dataset.
        geojson_path: Path to the hotspots GeoJSON file.
        forecast_dir: Directory containing Prophet forecast CSVs.
        config_path: Path to params.yaml config.

    Returns:
        Configured Dash app instance.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dash_cfg = config.get("dashboard", {})
    title = dash_cfg.get("title", "Epidemiological Pulse")

    loader = DashboardDataLoader(
        master_csv_path=master_csv_path,
        geojson_path=geojson_path,
        forecast_dir=forecast_dir,
        config=config,
    )
    charts = ChartBuilder()

    # Pre-load data
    try:
        master_df = loader.load_master()
        all_zipcodes = loader.get_all_zipcodes()
        date_min, date_max = loader.get_date_range()
        default_end = date_max
        default_start = default_end - timedelta(days=dash_cfg.get("default_date_range_days", 90))
    except Exception as exc:
        logger.error(f"Failed to load dashboard data: {exc}")
        master_df = pd.DataFrame()
        all_zipcodes = []
        date_min = datetime(2022, 1, 1)
        date_max = datetime.now()
        default_start = date_max - timedelta(days=90)
        default_end = date_max

    # ---------------------------------------------------------------------------
    # App Layout
    # ---------------------------------------------------------------------------
    app = Dash(
        __name__,
        title=title,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@300;400;600;700&display=swap"
        ],
        suppress_callback_exceptions=True,
    )

    def card(children, style_extra: dict = None) -> html.Div:
        style = {
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "12px",
            "padding": "20px",
            "marginBottom": "16px",
        }
        if style_extra:
            style.update(style_extra)
        return html.Div(children, style=style)

    def metric_card(label: str, value: str, color: str = COLORS["accent_cyan"]) -> html.Div:
        return html.Div(
            [
                html.Div(label, style={"color": COLORS["text_secondary"], "fontSize": "12px", "fontWeight": "600", "letterSpacing": "1px", "textTransform": "uppercase"}),
                html.Div(value, style={"color": color, "fontSize": "32px", "fontWeight": "700", "fontFamily": "DM Mono, monospace", "marginTop": "4px"}),
            ],
            style={
                "background": COLORS["card"],
                "border": f"1px solid {COLORS['card_border']}",
                "borderLeft": f"4px solid {color}",
                "borderRadius": "8px",
                "padding": "16px 20px",
                "flex": "1",
                "minWidth": "140px",
            },
        )

    # Summary metrics
    if not master_df.empty:
        recent_df = master_df[master_df["date"] >= master_df["date"].max() - pd.Timedelta(days=30)]
        avg_er = f"{recent_df['er_visits'].mean():.0f}"
        n_high = (
            loader.get_region_summary()["risk_level"].value_counts().get("High", 0)
            if not master_df.empty else 0
        )
        avg_aqi = f"{recent_df['aqi'].mean():.0f}" if "aqi" in recent_df.columns else "N/A"
        avg_sentiment = f"{recent_df['negative_sentiment'].mean():.2%}" if "negative_sentiment" in recent_df.columns else "N/A"
    else:
        avg_er = "N/A"
        n_high = 0
        avg_aqi = "N/A"
        avg_sentiment = "N/A"

    app.layout = html.Div(
        style={
            "background": COLORS["background"],
            "minHeight": "100vh",
            "fontFamily": "Space Grotesk, sans-serif",
            "color": COLORS["text_primary"],
            "padding": "20px 32px",
        },
        children=[
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "🔬 Epidemiological Pulse",
                                style={"margin": 0, "fontSize": "26px", "fontWeight": "700"},
                            ),
                            html.P(
                                "Population Health Hotspot Detection & ER Visit Forecasting",
                                style={"margin": "4px 0 0", "color": COLORS["text_secondary"], "fontSize": "13px"},
                            ),
                        ]
                    ),
                    html.Div(
                        html.Span(
                            "● LIVE",
                            style={"color": COLORS["risk_low"], "fontSize": "12px", "fontWeight": "600"},
                        ),
                        style={"alignSelf": "center"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "24px",
                    "borderBottom": f"1px solid {COLORS['card_border']}",
                    "paddingBottom": "20px",
                },
            ),

            # KPI Row
            html.Div(
                [
                    metric_card("Avg Daily ER Visits", avg_er, COLORS["accent_cyan"]),
                    metric_card("High Risk Regions", str(n_high), COLORS["risk_high"]),
                    metric_card("Avg AQI (30d)", avg_aqi, COLORS["accent_blue"]),
                    metric_card("Neg Sentiment (30d)", avg_sentiment, COLORS["risk_medium"]),
                ],
                style={"display": "flex", "gap": "12px", "marginBottom": "20px", "flexWrap": "wrap"},
            ),

            # Filter Row
            card(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Region (ZIP)", style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginBottom": "6px"}),
                                dcc.Dropdown(
                                    id="zip-filter",
                                    options=[{"label": z, "value": z} for z in all_zipcodes],
                                    value=all_zipcodes[0] if all_zipcodes else None,
                                    clearable=False,
                                    style={"background": COLORS["card"], "color": "#000"},
                                ),
                            ],
                            style={"flex": "1", "minWidth": "150px"},
                        ),
                        html.Div(
                            [
                                html.Label("Date Range", style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginBottom": "6px"}),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    start_date=default_start.date(),
                                    end_date=default_end.date(),
                                    display_format="YYYY-MM-DD",
                                    style={"fontSize": "13px"},
                                ),
                            ],
                            style={"flex": "2"},
                        ),
                        html.Div(
                            [
                                html.Label("Risk Filter", style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginBottom": "6px"}),
                                dcc.Checklist(
                                    id="risk-filter",
                                    options=[
                                        {"label": " 🔴 High", "value": "High"},
                                        {"label": " 🟠 Medium", "value": "Medium"},
                                        {"label": " 🟢 Low", "value": "Low"},
                                    ],
                                    value=["High", "Medium", "Low"],
                                    inline=True,
                                    labelStyle={"marginRight": "12px", "fontSize": "13px"},
                                ),
                            ],
                            style={"flex": "2"},
                        ),
                    ],
                    style={"display": "flex", "gap": "24px", "alignItems": "flex-end", "flexWrap": "wrap"},
                )
            ),

            # Map + Risk Distribution Row
            html.Div(
                [
                    html.Div(
                        card(dcc.Graph(id="hotspot-map", config={"displayModeBar": False})),
                        style={"flex": "3"},
                    ),
                    html.Div(
                        card(dcc.Graph(id="risk-distribution-bar", config={"displayModeBar": False})),
                        style={"flex": "1", "minWidth": "240px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px"},
            ),

            # ER Visits Forecast Chart
            card(dcc.Graph(id="er-timeseries", config={"displayModeBar": "hover"})),

            # Signals Chart
            card(dcc.Graph(id="signals-timeseries", config={"displayModeBar": False})),

            # Risk Score Heatmap
            card(dcc.Graph(id="risk-heatmap", config={"displayModeBar": False})),

            # Auto-refresh interval
            dcc.Interval(
                id="auto-refresh",
                interval=dash_cfg.get("refresh_interval_seconds", 300) * 1000,
                n_intervals=0,
            ),
        ],
    )

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------

    @app.callback(
        Output("hotspot-map", "figure"),
        Output("risk-distribution-bar", "figure"),
        Output("risk-heatmap", "figure"),
        Input("risk-filter", "value"),
        Input("auto-refresh", "n_intervals"),
    )
    def update_map_and_summary(risk_filter_values, _n):
        try:
            region_summary = loader.get_region_summary(window_days=30)
            if risk_filter_values:
                region_summary = region_summary[
                    region_summary["risk_level"].isin(risk_filter_values)
                ]
            map_fig = charts.hotspot_map(region_summary)
            risk_bar = charts.risk_distribution_bar(loader.get_region_summary(30))
            heatmap = charts.risk_score_heatmap(loader.load_master())
            return map_fig, risk_bar, heatmap
        except Exception as exc:
            logger.error(f"Map update error: {exc}")
            empty = go.Figure()
            return empty, empty, empty

    @app.callback(
        Output("er-timeseries", "figure"),
        Output("signals-timeseries", "figure"),
        Input("zip-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    )
    def update_timeseries(zipcode, start_date, end_date):
        if not zipcode:
            raise PreventUpdate

        try:
            ts_df = loader.get_region_timeseries(zipcode, start_date, end_date)
            forecast_df = loader.load_prophet_forecast(zipcode)

            er_fig = charts.er_timeseries(ts_df, forecast_df, zipcode)
            signals_fig = charts.signal_timeseries(ts_df, zipcode)

            return er_fig, signals_fig
        except Exception as exc:
            logger.error(f"Timeseries update error: {exc}")
            empty = go.Figure()
            return empty, empty

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_dashboard(
    config_path: str = "config/params.yaml",
    data_dir: str = "data/synthetic",
    debug: bool = False,
) -> None:
    """
    Launch the Dash dashboard.

    Args:
        config_path: Path to YAML config.
        data_dir: Directory containing master_dataset.csv.
        debug: Run in debug mode.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dash_cfg = cfg.get("dashboard", {})

    master_csv = os.path.join(data_dir, "master_dataset.csv")
    geojson_path = cfg.get("clustering", {}).get("geojson_output", "outputs/geojson/hotspots.geojson")
    forecast_dir = cfg.get("prophet", {}).get("model_save_dir", "outputs/models/prophet")

    app = create_app(
        master_csv_path=master_csv,
        geojson_path=geojson_path,
        forecast_dir=forecast_dir,
        config_path=config_path,
    )

    logger.info(
        f"Starting dashboard at http://{dash_cfg.get('host', '0.0.0.0')}:"
        f"{dash_cfg.get('port', 8050)}"
    )

    app.run(
        host=dash_cfg.get("host", "0.0.0.0"),
        port=dash_cfg.get("port", 8050),
        debug=debug,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard(debug=True)
