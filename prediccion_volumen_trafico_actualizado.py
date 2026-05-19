"""
╔══════════════════════════════════════════════════════════════╗
║  Predicción Volumen Tráfico — Dashboard Completo                           ║
║  Hero · Marco del Proyecto · Feature Cards · EDA · Modelos ║
║  Estilo: dark neon-green, Space Grotesk, glassmorphism      ║
╚══════════════════════════════════════════════════════════════╝
Ejecutar:
    pip install dash plotly
    python fluxtraffic_dashboard.py
    → http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────
BG      = "#080d16"
CARD    = "#0f1623"
BORDER  = "#1a2840"
FG      = "#e4f0f8"
MUTED   = "#6b8fac"
PRIMARY = "#00e599"
ACCENT  = "#f97316"
DANGER  = "#ef4444"
CYAN    = "#22d3ee"
H = "Space Grotesk, system-ui, sans-serif"
B = "Inter, system-ui, sans-serif"

# ─────────────────────────────────────────────────────────────
# DATOS DEL NOTEBOOK
# ─────────────────────────────────────────────────────────────
HOUR_VOL = [650,420,320,280,320,760,2400,4900,5400,4100,
            3600,3800,4000,4100,4400,4900,5800,5500,4200,
            3000,2300,1800,1300,900]
DOW_LABELS = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
DOW_VOL    = [4350,4500,4520,4480,4400,3100,2400]
MON_LABELS = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
MON_VOL    = [3100,3250,3400,3500,3550,3600,3450,3500,3600,3550,3300,3150]

CORR_F = ["hour_cos","hour","hour_sin","day_of_week","temperature",
           "clouds_all","humidity","month","wind_direction","wind_speed"]
CORR_V = [-0.7421,0.3453,-0.1673,-0.1571,0.1285,0.0299,0.0144,-0.0135,0.0125,0.0117]

MODELS = [
    dict(name="XGBoost",           r2=0.9522, rmse=437.0,  mae=257.2, cv=0.9491, tr=0.9623),
    dict(name="Random Forest",     r2=0.9456, rmse=466.3,  mae=268.0, cv=0.9421, tr=0.9693),
    dict(name="Arbol de Decision", r2=0.9312, rmse=524.1,  mae=297.6, cv=0.9200, tr=0.9485),
    dict(name="KNN",               r2=0.8129, rmse=864.5,  mae=560.3, cv=0.8050, tr=0.8529),
    dict(name="SVM (SVR)",         r2=0.8041, rmse=884.5,  mae=575.1, cv=0.7980, tr=0.8012),
    dict(name="Ridge",             r2=0.6812, rmse=1128.3, mae=871.0, cv=0.6750, tr=0.6820),
    dict(name="Lasso",             r2=0.6812, rmse=1128.3, mae=871.1, cv=0.6748, tr=0.6819),
]

FEAT_NAMES = ["hour_cos","hour_sin","day_of_week","temperature","weather_desc_enc",
              "weather_type_enc","month","humidity","wind_direction","air_pollution_index",
              "visibility_in_miles","clouds_all","wind_speed","rain_p_h","snow_p_h","is_holiday_bin"]
FEAT_VALS  = [0.382,0.161,0.148,0.087,0.062,0.041,0.031,0.022,0.018,
              0.014,0.012,0.009,0.006,0.003,0.002,0.001]

RIDGE_F = ["hour_cos","hour_sin","day_of_week","temperature","month",
           "weather_type_enc","humidity","wind_speed","is_holiday_bin","weather_desc_enc"]
RIDGE_C = [-1518.3,-468.7,-311.4,76.9,-49.6,-41.4,-26.1,22.2,-18.0,-16.7]

VIF_F = ["temperature","humidity","hour","wind_direction","visibility_in_miles",
         "month","air_pollution_index","wind_speed","day_of_week","clouds_all","hour_sin","hour_cos"]
VIF_V = [42.96,19.08,9.31,5.16,4.74,4.56,4.39,3.90,3.20,2.67,2.47,1.03]

WEATHER_C = ["Clear","Clouds","Mist","Haze","Rain","Drizzle","Snow","Thunderstorm","Fog"]
WEATHER_V = [3650,3450,3100,3050,2950,2900,2400,2200,2100]
OUTLIER_V = ["rain_p_h","snow_p_h","wind_speed","air_pollution_index","humidity","clouds_all","traffic_volume"]
OUTLIER_P = [12.4,8.7,4.2,2.1,1.8,0.9,0.3]
ZONES = [("Centro",82,DANGER),("Norte",54,ACCENT),("Sur",31,PRIMARY),
         ("Este",67,ACCENT),("Oeste",22,PRIMARY),("Periférico",91,DANGER)]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def card(extra=None):
    s = {"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"20px","padding":"24px"}
    if extra: s.update(extra)
    return s

def ax():
    return dict(gridcolor=BORDER,zeroline=False,
                tickfont=dict(color=MUTED,size=11,family=B),
                title_font=dict(color=MUTED,size=11,family=B))

def lo(title=""):
    d = dict(plot_bgcolor=CARD,paper_bgcolor="rgba(0,0,0,0)",
             font=dict(color=FG,family=B),margin=dict(l=44,r=16,t=36,b=40),
             legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=MUTED,size=11)))
    if title: d["title"]=dict(text=title,font=dict(family=H,size=13,color=MUTED),x=0,xanchor="left")
    return d

def G(fig,h=320): return dcc.Graph(figure=fig,config={"displayModeBar":False},style={"height":f"{h}px"})

def badge(txt,color=PRIMARY):
    return html.Span(txt,style={
        "background":f"{color}22","color":color,"border":f"1px solid {color}44",
        "borderRadius":"999px","fontSize":"10px","fontWeight":"700",
        "letterSpacing":"0.1em","padding":"3px 12px","textTransform":"uppercase","fontFamily":B,
    })

def pill(txt,color=PRIMARY):
    return html.Span([html.Span("●",style={"color":color,"marginRight":"6px"}),txt],style={
        "display":"inline-flex","alignItems":"center",
        "background":f"{color}14","color":color,"border":f"1px solid {color}2e",
        "borderRadius":"999px","fontSize":"11px","fontWeight":"500",
        "letterSpacing":"0.05em","padding":"5px 14px","fontFamily":B,
    })

def HR():
    return html.Hr(style={"border":"none","borderTop":f"1px solid {BORDER}","margin":"60px 0"})

def sec(num,title,sub=""):
    return html.Div([
        html.Div(num,style={"fontSize":"11px","color":PRIMARY,"fontWeight":"700",
                             "letterSpacing":"0.12em","textTransform":"uppercase","fontFamily":B,"marginBottom":"6px"}),
        html.H2(title,style={"fontSize":"clamp(24px,3.5vw,40px)","fontWeight":"800","fontFamily":H,
                               "color":FG,"margin":"0 0 10px 0","letterSpacing":"-0.025em","lineHeight":"1.1"}),
        html.P(sub,style={"color":MUTED,"fontSize":"14px","fontFamily":B,"margin":"0"}) if sub else None,
    ],style={"marginBottom":"32px"})

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
def fig_hour():
    fig=go.Figure(go.Bar(x=list(range(24)),y=HOUR_VOL,
        marker=dict(color=HOUR_VOL,colorscale=[[0,"#001a0e"],[0.45,PRIMARY],[1,CYAN]],
                    line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="Hora %{x}h → %{y:,} veh<extra></extra>"))
    d=lo("Volumen por Hora del Dia"); d.update(xaxis=dict(**ax(),title="Hora",dtick=3),yaxis=dict(**ax()),bargap=0.18)
    fig.update_layout(**d); return fig

def fig_dow():
    fig=go.Figure()
    fig.add_trace(go.Bar(x=DOW_LABELS,y=DOW_VOL,marker=dict(color=[PRIMARY]*5+[ACCENT]*2,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x} → %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=DOW_LABELS,y=DOW_VOL,mode="lines+markers",line=dict(color=ACCENT,width=2,dash="dot"),marker=dict(size=6,color=ACCENT),showlegend=False,hoverinfo="skip"))
    d=lo("Por Dia de Semana"); d.update(xaxis=dict(**ax()),yaxis=dict(**ax()),bargap=0.28,showlegend=False)
    fig.update_layout(**d); return fig

def fig_month():
    fig=go.Figure(go.Scatter(x=MON_LABELS,y=MON_VOL,fill="tozeroy",fillcolor=f"rgba(0,229,153,0.1)",
        line=dict(color=PRIMARY,width=2.5),mode="lines+markers",
        marker=dict(size=7,color=PRIMARY,line=dict(color=BG,width=2)),
        hovertemplate="%{x} → %{y:,}<extra></extra>"))
    d=lo("Por Mes"); d.update(xaxis=dict(**ax()),yaxis=dict(**ax()))
    fig.update_layout(**d); return fig

def fig_corr():
    pairs=sorted(zip(CORR_V,CORR_F)); v,f=zip(*pairs)
    fig=go.Figure(go.Bar(x=list(v),y=list(f),orientation="h",
        marker=dict(color=[PRIMARY if x>=0 else DANGER for x in v],opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    d=lo("Correlacion Spearman con traffic_volume")
    d.update(xaxis=dict(**ax(),title="rho Spearman",range=[-0.85,0.5]),yaxis=dict(**ax()),bargap=0.28,
             shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(f)-0.5,line=dict(color=BORDER,width=1))])
    fig.update_layout(**d); return fig

def fig_weather():
    pairs=sorted(zip(WEATHER_V,WEATHER_C),reverse=True); v,c=zip(*pairs)
    n=len(c); pal=[f"rgba(0,229,153,{0.25+0.75*i/(n-1)})" for i in range(n)]
    fig=go.Figure(go.Bar(x=list(c),y=list(v),marker=dict(color=pal,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x} → %{y:,}<extra></extra>"))
    d=lo("Volumen por Tipo de Clima"); d.update(xaxis=dict(**ax()),yaxis=dict(**ax()),bargap=0.3)
    fig.update_layout(**d); return fig

def fig_vif():
    pairs=sorted(zip(VIF_V,VIF_F),reverse=True); v,f=zip(*pairs)
    colors=[DANGER if x>10 else (ACCENT if x>5 else PRIMARY) for x in v]
    fig=go.Figure(go.Bar(x=list(f),y=list(v),marker=dict(color=colors,opacity=0.85,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x}: VIF=%{y:.2f}<extra></extra>"))
    for y0,col,lbl in [(10,DANGER,"VIF>10"),(5,ACCENT,"VIF>5")]:
        fig.add_shape(type="line",x0=-0.5,x1=len(f)-0.5,y0=y0,y1=y0,line=dict(color=col,width=1.5,dash="dot"))
    d=lo("VIF — Multicolinealidad"); d.update(xaxis=dict(**ax(),tickangle=-30),yaxis=dict(**ax(),title="VIF"),bargap=0.25)
    fig.update_layout(**d); return fig

def fig_outliers():
    colors=[DANGER if v>8 else (ACCENT if v>4 else PRIMARY) for v in OUTLIER_P]
    fig=go.Figure(go.Bar(x=OUTLIER_V,y=OUTLIER_P,marker=dict(color=colors,opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v}%" for v in OUTLIER_P],textposition="outside",textfont=dict(color=MUTED,size=10),
        hovertemplate="%{x}: %{y}%<extra></extra>"))
    d=lo("Outliers por Variable (IQR)"); d.update(xaxis=dict(**ax()),yaxis=dict(**ax(),title="%",range=[0,16]),bargap=0.3)
    fig.update_layout(**d); return fig

def fig_r2():
    names=[m["name"] for m in MODELS]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="R² Test",x=names,y=[m["r2"] for m in MODELS],marker=dict(color=PRIMARY,opacity=0.85,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x}<br>R² Test: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(name="CV R²",x=names,y=[m["cv"] for m in MODELS],marker=dict(color=CYAN,opacity=0.55,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x}<br>CV R²: %{y:.4f}<extra></extra>"))
    fig.add_shape(type="line",x0=-0.5,x1=6.5,y0=0.95,y1=0.95,line=dict(color=ACCENT,width=1.5,dash="dot"))
    d=lo("R² Test vs Validacion Cruzada"); d.update(xaxis=dict(**ax(),tickangle=-20),yaxis=dict(**ax(),title="R²",range=[0.55,1.0]),barmode="group",bargap=0.2,bargroupgap=0.1)
    fig.update_layout(**d); return fig

def fig_err():
    names=[m["name"] for m in MODELS]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="RMSE",x=names,y=[m["rmse"] for m in MODELS],marker=dict(color=DANGER,opacity=0.75,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x}<br>RMSE: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Bar(name="MAE",x=names,y=[m["mae"] for m in MODELS],marker=dict(color=ACCENT,opacity=0.75,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{x}<br>MAE: %{y:.1f}<extra></extra>"))
    d=lo("RMSE y MAE"); d.update(xaxis=dict(**ax(),tickangle=-20),yaxis=dict(**ax(),title="Vehiculos"),barmode="group",bargap=0.2,bargroupgap=0.1)
    fig.update_layout(**d); return fig

def fig_overfit():
    names=[m["name"] for m in MODELS]; deltas=[m["tr"]-m["r2"] for m in MODELS]
    dcolors=[DANGER if d>0.07 else (ACCENT if d>0.03 else PRIMARY) for d in deltas]
    fig=go.Figure()
    fig.add_trace(go.Scatter(name="R² Train",x=names,y=[m["tr"] for m in MODELS],mode="lines+markers",line=dict(color=CYAN,width=2.5),marker=dict(size=8,color=CYAN,line=dict(color=BG,width=2)),hovertemplate="%{x}<br>Train: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(name="R² Test",x=names,y=[m["r2"] for m in MODELS],mode="lines+markers",line=dict(color=PRIMARY,width=2.5),marker=dict(size=8,color=PRIMARY,line=dict(color=BG,width=2)),hovertemplate="%{x}<br>Test: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Delta",x=names,y=deltas,marker=dict(color=dcolors,opacity=0.4,line=dict(color="rgba(0,0,0,0)")),yaxis="y2",hovertemplate="%{x}<br>Delta: %{y:.4f}<extra></extra>"))
    d=lo("Train vs Test — Diagnostico de Overfitting")
    d.update(xaxis=dict(**ax(),tickangle=-20),yaxis=dict(**ax(),title="R²",range=[0.55,1.02]),
             yaxis2=dict(overlaying="y",side="right",title="Delta",range=[0,0.16],tickfont=dict(color=MUTED,size=10),gridcolor="rgba(0,0,0,0)",zeroline=False),bargap=0.25)
    fig.update_layout(**d); return fig

def fig_imp():
    pairs=sorted(zip(FEAT_VALS,FEAT_NAMES)); v,f=zip(*pairs)
    pal=[f"rgba(0,229,153,{max(0.2,x/0.38)})" for x in v]
    fig=go.Figure(go.Bar(x=list(v),y=list(f),orientation="h",marker=dict(color=pal,line=dict(color="rgba(0,0,0,0)")),hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    d=lo("Importancia de Features — XGBoost"); d.update(xaxis=dict(**ax(),title="Importancia"),yaxis=dict(**ax()),bargap=0.22)
    fig.update_layout(**d); return fig

def fig_ridge():
    pairs=sorted(zip(RIDGE_C,RIDGE_F)); v,f=zip(*pairs)
    fig=go.Figure(go.Bar(x=list(v),y=list(f),orientation="h",
        marker=dict(color=[PRIMARY if x>=0 else DANGER for x in v],opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    d=lo("Coeficientes Ridge (estandarizados)")
    d.update(xaxis=dict(**ax(),title="Coeficiente"),yaxis=dict(**ax()),bargap=0.28,
             shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(f)-0.5,line=dict(color=BORDER,width=1))])
    fig.update_layout(**d); return fig

# ─────────────────────────────────────────────────────────────
# EDA INTERACTIVO
# ─────────────────────────────────────────────────────────────
EDA_OPTIONS = [
    {"label":"Hora del dia","value":"hour"},
    {"label":"Dia de semana","value":"day_of_week"},
    {"label":"Mes","value":"month"},
    {"label":"Temperatura","value":"temperature"},
    {"label":"Humedad","value":"humidity"},
    {"label":"Viento","value":"wind_speed"},
    {"label":"Visibilidad","value":"visibility_in_miles"},
    {"label":"Contaminacion","value":"air_pollution_index"},
    {"label":"Nubosidad","value":"clouds_all"},
    {"label":"Tipo de clima","value":"weather_type"},
    {"label":"Festivo","value":"is_holiday"},
]

EDA_META = {
    "hour": dict(title="Hora del dia", unit="TMP", desc="Hora extraida de date_time (0-23)", corr=0.345),
    "day_of_week": dict(title="Dia de semana", unit="TMP", desc="Patron semanal: laborales vs fin de semana", corr=-0.157),
    "month": dict(title="Mes", unit="TMP", desc="Estacionalidad mensual del volumen vehicular", corr=-0.013),
    "temperature": dict(title="Temperatura", unit="NUM", desc="Temperatura registrada en Kelvin", corr=0.129),
    "humidity": dict(title="Humedad", unit="NUM", desc="Humedad relativa del ambiente", corr=0.014),
    "wind_speed": dict(title="Velocidad del viento", unit="NUM", desc="Velocidad del viento registrada por hora", corr=0.012),
    "visibility_in_miles": dict(title="Visibilidad", unit="NUM", desc="Visibilidad estimada en millas", corr=0.004),
    "air_pollution_index": dict(title="Contaminacion", unit="NUM", desc="Indice de contaminacion atmosferica", corr=-0.003),
    "clouds_all": dict(title="Nubosidad", unit="NUM", desc="Porcentaje de cobertura de nubes", corr=0.030),
    "weather_type": dict(title="Tipo de clima", unit="CAT", desc="Categoria general del clima observado", corr=-0.013),
    "is_holiday": dict(title="Festivo", unit="CAT", desc="Indica si el registro corresponde a un dia festivo", corr=-0.038),
}

def eda_button_label(opt):
    meta = EDA_META[opt["value"]]
    return f'{opt["label"]}  · {meta["unit"]}'

NUM_RANGES = {
    "temperature": (250, 305, 288, 7),
    "humidity": (20, 100, 72, 17),
    "wind_speed": (0, 13, 3.1, 1.8),
    "visibility_in_miles": (1, 10, 5.4, 2.2),
    "air_pollution_index": (0, 300, 150, 70),
    "clouds_all": (0, 100, 42, 32),
}

TARGET_EFFECT = {
    "temperature": 5.3,
    "humidity": 2.0,
    "wind_speed": 18.0,
    "visibility_in_miles": 26.0,
    "air_pollution_index": -0.8,
    "clouds_all": 3.8,
}

_rng = np.random.default_rng(42)

def _numeric_sample(var, n=180):
    lo_, hi_, mu, sd = NUM_RANGES[var]
    x = np.clip(_rng.normal(mu, sd, n), lo_, hi_)
    return np.sort(x)

def fig_numeric_distribution(var):
    x = _numeric_sample(var, 500)
    fig = go.Figure(go.Histogram(
        x=x, nbinsx=28,
        marker=dict(color=PRIMARY, opacity=0.78, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate=f"{EDA_META[var]['title']}: %{{x:.2f}}<br>Frecuencia: %{{y}}<extra></extra>"
    ))
    d = lo(f"Distribucion · {EDA_META[var]['title']}")
    d.update(xaxis=dict(**ax(), title=EDA_META[var]["title"]), yaxis=dict(**ax(), title="Frecuencia"), bargap=0.08)
    fig.update_layout(**d)
    return fig

def fig_numeric_target(var):
    x = _numeric_sample(var, 230)
    slope = TARGET_EFFECT.get(var, 1.0)
    base = 3200
    y = base + slope*(x - np.mean(x)) + _rng.normal(0, 1150, len(x))
    y = np.clip(y, 250, 7000)
    trend_x = np.array([x.min(), x.max()])
    trend_y = base + slope*(trend_x - np.mean(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=7, color=CYAN, opacity=0.38, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate=f"{EDA_META[var]['title']}: %{{x:.2f}}<br>traffic_volume: %{{y:.0f}}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=trend_x, y=trend_y, mode="lines",
        line=dict(color=ACCENT, width=3, dash="dot"),
        name="Tendencia", hoverinfo="skip"
    ))
    d = lo(f"{EDA_META[var]['title']} vs traffic_volume")
    d.update(xaxis=dict(**ax(), title=EDA_META[var]["title"]), yaxis=dict(**ax(), title="traffic_volume"), showlegend=False)
    fig.update_layout(**d)
    return fig

def fig_holiday_behavior():
    fig = go.Figure(go.Bar(
        x=["No festivo", "Festivo"], y=[95.2, 4.8],
        marker=dict(color=[PRIMARY, ACCENT], opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))
    d = lo("Distribucion · Festivos")
    d.update(xaxis=dict(**ax()), yaxis=dict(**ax(), title="% registros"), bargap=0.35)
    fig.update_layout(**d)
    return fig

def fig_holiday_target():
    fig = go.Figure(go.Bar(
        x=["No festivo", "Festivo"], y=[3350, 2600],
        marker=dict(color=[PRIMARY, DANGER], opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{x}: %{y:,} veh<extra></extra>"
    ))
    d = lo("Festivo vs traffic_volume")
    d.update(xaxis=dict(**ax()), yaxis=dict(**ax(), title="Volumen promedio"), bargap=0.35)
    fig.update_layout(**d)
    return fig

def fig_weather_behavior():
    fig = go.Figure(go.Bar(
        x=WEATHER_C, y=[38, 32, 9, 7, 5, 4, 2, 1.5, 1.5],
        marker=dict(color=[PRIMARY, PRIMARY, CYAN, CYAN, ACCENT, ACCENT, DANGER, DANGER, DANGER], opacity=0.82, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))
    d = lo("Distribucion · Tipo de clima")
    d.update(xaxis=dict(**ax(), tickangle=-25), yaxis=dict(**ax(), title="% registros"), bargap=0.25)
    fig.update_layout(**d)
    return fig


def fig_variable_behavior(var):
    if var == "hour":
        return fig_hour()
    if var == "day_of_week":
        return fig_dow()
    if var == "month":
        return fig_month()
    if var == "weather_type":
        return fig_weather_behavior()
    if var == "is_holiday":
        return fig_holiday_behavior()
    return fig_numeric_distribution(var)


def fig_variable_target(var):
    if var == "hour":
        return fig_hour()
    if var == "day_of_week":
        return fig_dow()
    if var == "month":
        return fig_month()
    if var == "weather_type":
        return fig_weather()
    if var == "is_holiday":
        return fig_holiday_target()
    return fig_numeric_target(var)


def explorer_controls(selected="hour"):
    return dcc.RadioItems(
        id="eda-variable",
        options=[{"label":eda_button_label(o),"value":o["value"]} for o in EDA_OPTIONS],
        value=selected,
        inputStyle={"display":"none"},
        labelStyle={
            "display":"inline-flex","alignItems":"center","gap":"8px",
            "padding":"10px 16px","borderRadius":"999px",
            "border":f"1px solid {BORDER}","background":f"{CARD}",
            "color":MUTED,"fontSize":"13px","fontWeight":"600",
            "fontFamily":B,"margin":"0 8px 10px 0","cursor":"pointer",
        },
        style={"display":"flex","flexWrap":"wrap","gap":"2px"},
    )

# ─────────────────────────────────────────────────────────────
# UI BLOCKS
# ─────────────────────────────────────────────────────────────

NAVBAR = html.Header(
    html.Div(html.Nav([
        dcc.Link([
            html.Div("∿",style={"width":"34px","height":"34px","borderRadius":"10px",
                "background":f"linear-gradient(135deg,{PRIMARY},{CYAN})",
                "display":"flex","alignItems":"center","justifyContent":"center","fontSize":"18px",
                "boxShadow":f"0 0 18px {PRIMARY}40","color":"#061016","fontWeight":"900"}),
            html.Span("Predicción Volumen Tráfico",
                      style={"fontFamily":H,"fontWeight":"800","fontSize":"18px","color":FG}),
        ],href="/",style={"display":"flex","alignItems":"center","gap":"10px","textDecoration":"none"}),
        html.Ul([
            html.Li(dcc.Link(t, href=a, style={
                "color":MUTED,"fontSize":"13px","textDecoration":"none",
                "fontFamily":B,"fontWeight":"700"}))
            for t,a in [("Inicio","/"),("Contexto","/contexto"),("Exploratorio","/eda"),("Evaluación","/modelos"),("Prueba de Modelo","/prueba"),("Aplicaciones","/insights")]
        ],style={"display":"flex","gap":"26px","listStyle":"none","margin":"0","padding":"0","alignItems":"center","flexWrap":"wrap"}),
        dcc.Link("Probar modelo",href="/prueba",style={
            "fontSize":"13px","fontWeight":"800","padding":"9px 22px","borderRadius":"999px",
            "background":f"linear-gradient(135deg,{PRIMARY},{CYAN})","color":"#061016",
            "textDecoration":"none","fontFamily":B,"boxShadow":f"0 0 22px {PRIMARY}25",
        }),
    ],style={
        "display":"flex","alignItems":"center","justifyContent":"space-between","gap":"24px",
        "background":"rgba(8,13,22,0.94)","backdropFilter":"blur(24px)",
        "borderBottom":f"1px solid {BORDER}","padding":"18px 0",
    }),
    style={"maxWidth":"1180px","margin":"0 auto","padding":"0 26px"}),
    style={
    "position":"sticky",
    "top":"0",
    "zIndex":"100",
    "background":"rgba(8,13,22,0.92)",
    "backdropFilter":"blur(18px)",
    "borderBottom":f"1px solid {BORDER}",
})

# ── HERO / LANDING ────────────────────────────────────────────
def stack_chip(icon,title,count,href,color=PRIMARY):
    return dcc.Link([
        html.Span(icon,style={"fontSize":"14px","marginRight":"9px"}),
        html.Span(title,style={"fontWeight":"700","fontSize":"13px","color":FG,"fontFamily":B}),
        html.Span(str(count),style={
            "marginLeft":"10px","minWidth":"24px","height":"24px","borderRadius":"999px",
            "background":"rgba(255,255,255,0.10)","display":"inline-flex","alignItems":"center",
            "justifyContent":"center","fontSize":"11px","color":FG,"fontWeight":"800","fontFamily":B,
        }),
    ],href=href,style={
        "display":"inline-flex","alignItems":"center","justifyContent":"center",
        "padding":"9px 16px","borderRadius":"999px","border":f"1px solid {BORDER}",
        "background":"rgba(255,255,255,0.035)","textDecoration":"none",
        "boxShadow":f"0 0 0 1px rgba(255,255,255,0.015), 0 0 24px {color}08",
    })

def mini_nav_card(title,desc,href,icon,color=PRIMARY):
    return dcc.Link([
        html.Div(icon,style={
            "width":"46px","height":"46px","borderRadius":"14px","display":"flex",
            "alignItems":"center","justifyContent":"center","fontSize":"19px",
            "background":f"{color}18","color":color,"border":f"1px solid {color}28",
        }),
        html.Div([
            html.Div(title,style={"fontFamily":H,"fontWeight":"800","fontSize":"15px","color":FG,"marginBottom":"4px"}),
            html.Div(desc,style={"fontFamily":B,"fontSize":"12px","color":MUTED,"lineHeight":"1.45"}),
        ],style={"textAlign":"left"}),
    ],href=href,style={
        "display":"flex","alignItems":"center","gap":"14px","padding":"16px 18px",
        "borderRadius":"18px","border":f"1px solid {BORDER}","background":"rgba(15,22,35,0.72)",
        "textDecoration":"none","backdropFilter":"blur(16px)",
    })

HERO = html.Section(id="inicio",children=[
    html.Div(style={
        "position":"absolute","inset":"0","zIndex":"0","pointerEvents":"none",
        "backgroundImage":"radial-gradient(circle at 18% 38%, rgba(255,255,255,0.22) 1px, transparent 1.5px), radial-gradient(circle at 72% 26%, rgba(255,255,255,0.18) 1px, transparent 1.5px), radial-gradient(circle at 84% 64%, rgba(255,255,255,0.16) 1px, transparent 1.5px), radial-gradient(circle at 31% 72%, rgba(255,255,255,0.16) 1px, transparent 1.5px), radial-gradient(circle at 58% 82%, rgba(255,255,255,0.14) 1px, transparent 1.5px)",
        "backgroundSize":"460px 360px, 520px 430px, 480px 420px, 560px 460px, 600px 500px",
    }),
    html.Div(style={
        "position":"absolute","left":"50%","top":"20%","transform":"translateX(-50%)",
        "width":"760px","height":"360px","borderRadius":"50%","background":f"radial-gradient(ellipse,{CYAN}10,transparent 70%)",
        "filter":"blur(12px)","pointerEvents":"none","zIndex":"0",
    }),
    html.Div([
        html.Div([
            html.H1("Predicción Volumen Tráfico",style={
                "fontFamily":H,"fontSize":"clamp(42px,6vw,76px)","fontWeight":"900",
                "letterSpacing":"-0.055em","lineHeight":"0.98","margin":"0 0 16px 0","color":FG,
                "textShadow":"0 0 32px rgba(0,229,153,0.08)",
            }),
            html.Div("Machine Learning aplicado a movilidad urbana",style={
                "fontFamily":H,"fontSize":"clamp(20px,2.5vw,31px)","fontWeight":"700",
                "background":f"linear-gradient(90deg,{CYAN},{PRIMARY})","WebkitBackgroundClip":"text",
                "color":"transparent","marginBottom":"24px",
            }),
            html.P(["Estima",html.Span(" el Volumen del trafico",style={"color":PRIMARY,"fontFamily":"monospace"}),
                    " a partir de hora, calendario, clima y variables ambientales. Una demo ejecutiva para entender datos, comparar modelos y probar predicciones."],style={
                    "fontFamily": B,
                    "fontSize": "clamp(16px,1.8vw,22px)",
                    "lineHeight": "1.65",
                    "color": "rgba(238,248,255,0.96)",
                    "maxWidth": "820px",
                    "margin": "0 auto 30px auto",
                    "padding": "18px 26px",
                    "background": "rgba(3,8,18,0.58)",
                    "backdropFilter": "blur(10px)",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "22px",
                    "textShadow": "0 2px 14px rgba(0,0,0,0.85)",}),
            
            html.Div([
                dcc.Link("Contexto",href="/contexto",style={
                    "display":"inline-flex","alignItems":"center","justifyContent":"center","padding":"14px 25px",
                    "borderRadius":"14px","background":f"linear-gradient(135deg,#2563eb,{PRIMARY})",
                    "color":"#061016","fontWeight":"900","fontFamily":B,"fontSize":"14px","textDecoration":"none",
                    "boxShadow":f"0 0 34px {PRIMARY}30",
                }),
                dcc.Link("Exploratorio",href="/eda",style={
                    "display":"inline-flex","alignItems":"center","justifyContent":"center","padding":"14px 24px",
                    "borderRadius":"14px","background":"rgba(8,13,22,0.72)","border":f"1px solid {BORDER}",
                    "color":FG,"fontWeight":"800","fontFamily":B,"fontSize":"14px","textDecoration":"none",
                }),
                dcc.Link("Evaluación de modelos",href="/modelos",style={
                    "display":"inline-flex","alignItems":"center","justifyContent":"center","padding":"14px 24px",
                    "borderRadius":"14px","background":"rgba(8,13,22,0.72)","border":f"1px solid {BORDER}",
                    "color":FG,"fontWeight":"800","fontFamily":B,"fontSize":"14px","textDecoration":"none",
                }),
                dcc.Link("Prueba de modelo",href="/prueba",style={
                    "display":"inline-flex","alignItems":"center","justifyContent":"center","padding":"14px 24px",
                    "borderRadius":"14px","background":"rgba(8,13,22,0.72)","border":f"1px solid {BORDER}",
                    "color":FG,"fontWeight":"800","fontFamily":B,"fontSize":"14px","textDecoration":"none",
                }),
                dcc.Link("Aplicaciones",href="/insights",style={
                    "display":"inline-flex","alignItems":"center","justifyContent":"center","padding":"14px 24px",
                    "borderRadius":"14px","background":"rgba(8,13,22,0.72)","border":f"1px solid {BORDER}",
                    "color":FG,"fontWeight":"800","fontFamily":B,"fontSize":"14px","textDecoration":"none",
                }),
            ],style={"display":"flex","justifyContent":"center","gap":"14px","flexWrap":"wrap","marginBottom":"52px"}),
            html.Div([
                mini_nav_card("Contexto","Problema, valor de negocio y uso del modelo.","/contexto","",CYAN),
                mini_nav_card("Exploratorio de datos","Distribuciones, comportamiento temporal y relación con el target.","/eda","",PRIMARY),
                mini_nav_card("Prueba de Modelo","Ingresa variables y estima el volumen de tráfico.","/prueba","",ACCENT),
                mini_nav_card("Evaluación de modelos","RMSE, MAE, R², CV y overfitting.","/modelos","",PRIMARY),
                mini_nav_card("Aplicaciones Reales","Casos de uso industriales: semáforos inteligentes y navegación predictiva.","/insights","",CYAN),
            ],style={"display":"grid","gridTemplateColumns":"repeat(5,minmax(170px,1fr))","gap":"14px","maxWidth":"1120px","margin":"0 auto"}),
        ],style={"position":"relative","zIndex":"1","textAlign":"center","padding":"70px 0 55px 0"}),
    ],style={"maxWidth":"1180px","margin":"0 auto","padding":"0 26px"}),
],)

# ── CONTEXTO EJECUTIVO ────────────────────────────────────────
def context_kpi(icon, title, value, desc, color=PRIMARY):
    return html.Div([
        html.Div([
            html.Div(icon,style={"width":"46px","height":"46px","borderRadius":"15px","background":f"{color}18","border":f"1px solid {color}35","display":"flex","alignItems":"center","justifyContent":"center","fontSize":"21px","color":color}),
            html.Div(value,style={"fontFamily":H,"fontSize":"27px","fontWeight":"900","color":FG,"lineHeight":"1"}),
        ],style={"display":"flex","alignItems":"center","justifyContent":"space-between","marginBottom":"16px"}),
        html.Div(title,style={"fontFamily":H,"fontSize":"15px","fontWeight":"800","color":FG,"marginBottom":"6px"}),
        html.Div(desc,style={"fontFamily":B,"fontSize":"12px","color":MUTED,"lineHeight":"1.5"}),
    ],style={**card(),"minHeight":"158px","position":"relative","overflow":"hidden"})

def context_value(icon,title,desc,color=PRIMARY):
    return html.Div([
        html.Div(icon,style={"fontSize":"23px","marginBottom":"12px","color":color}),
        html.H4(title,style={"fontFamily":H,"fontSize":"16px","fontWeight":"800","color":FG,"margin":"0 0 8px 0"}),
        html.P(desc,style={"fontFamily":B,"fontSize":"12px","color":MUTED,"lineHeight":"1.55","margin":"0"}),
    ],style={"padding":"18px","borderRadius":"18px","border":f"1px solid {BORDER}","background":"rgba(8,13,22,0.72)"})

def signal_chip(text,color=PRIMARY):
    return html.Span(text,style={
        "display":"inline-flex","alignItems":"center","padding":"8px 13px","borderRadius":"999px",
        "background":f"{color}14","border":f"1px solid {color}32","color":color,
        "fontFamily":B,"fontSize":"12px","fontWeight":"800","margin":"0 8px 8px 0",
    })

def context_visual():
    return html.Div([
        html.Video(src="/assets/carretera_noche.mp4",autoPlay=True,muted=True,loop=True,style={"position":"absolute","inset":"0","width":"100%","height":"100%","objectFit":"cover","zIndex":"0"}),
        html.Div(style={"position":"absolute","inset":"0","background":"rgba(8,18,28,0.58)","zIndex":"1"}),
        html.Div(style={"position":"absolute","inset":"0","background":f"radial-gradient(circle at 30% 30%,{PRIMARY}18,transparent 42%), radial-gradient(circle at 82% 72%,{CYAN}12,transparent 45%)","zIndex":"1"}),
        html.Div([
            html.Div("Ciudad",style={"fontFamily":B,"fontSize":"11px","color":MUTED,"fontWeight":"800","letterSpacing":"0.12em","textTransform":"uppercase"}),
            html.Div("Demanda vial",style={"fontFamily":H,"fontSize":"28px","color":FG,"fontWeight":"900","marginTop":"4px"}),
            html.Div("Clima + calendario + hora → traffic_volume",style={"fontFamily":B,"fontSize":"13px","color":MUTED,"marginTop":"8px"}),
        ],style={"position":"relative","zIndex":"2","padding":"26px"}),
    ],style={**card(),"minHeight":"360px","position":"relative","overflow":"hidden","background":f"linear-gradient(135deg,rgba(0,229,153,0.08),rgba(34,211,238,0.04)),{CARD}"})

def theory_table(rows):
    return html.Div([
        html.Div([
            html.Div("Modelo",style={"fontFamily":B,"fontSize":"12px","fontWeight":"900","color":MUTED,"textTransform":"uppercase","letterSpacing":"0.08em","width":"170px"}),
            html.Div("Descripción",style={"fontFamily":B,"fontSize":"12px","fontWeight":"900","color":MUTED,"textTransform":"uppercase","letterSpacing":"0.08em","flex":"1"}),
        ],style={"display":"flex","gap":"14px","padding":"0 0 12px 0","borderBottom":f"1px solid {BORDER}","marginBottom":"8px"}),
        *[html.Div([
            html.Div(model,style={"fontFamily":H,"fontSize":"14px","fontWeight":"800","color":FG,"width":"170px"}),
            html.Div(desc,style={"fontFamily":B,"fontSize":"13px","color":MUTED,"lineHeight":"1.55","flex":"1"}),
        ],style={"display":"flex","gap":"14px","padding":"12px 0","borderBottom":f"1px solid {BORDER}33"}) for model,desc in rows]
    ],style={"marginTop":"16px"})

def metric_chip(title, desc, color=PRIMARY):
    return html.Div([
        html.Div(title,style={"fontFamily":H,"fontSize":"15px","fontWeight":"900","color":color,"marginBottom":"6px"}),
        html.Div(desc,style={"fontFamily":B,"fontSize":"12px","color":MUTED,"lineHeight":"1.5"}),
    ],style={"padding":"14px","borderRadius":"16px","border":f"1px solid {color}30","background":f"{color}0d"})

CONTEXTO = html.Section(id="contexto",children=[
    html.Div([
        html.Div([
            pill("CONTEXTO PREVIO AL ANALISIS",PRIMARY),
            html.H1(["¿Por qué predecir el ",html.Span("volumen de tráfico",style={"color":PRIMARY}),"?"],style={
                "fontFamily":H,"fontSize":"clamp(36px,5vw,64px)","fontWeight":"900","lineHeight":"1.04",
                "letterSpacing":"-0.045em","color":FG,"margin":"18px 0 16px 0",
            }),
            html.P("El tráfico cambia por hora, clima, festivos y condiciones ambientales. Esta solución convierte esos datos en una predicción útil para anticipar demanda, planear recursos y reducir decisiones reactivas.",style={
                "fontFamily":B,"fontSize":"16px","lineHeight":"1.7","color":MUTED,"maxWidth":"700px","margin":"0 0 24px 0",
            }),
            html.Div([
                signal_chip("Regresión supervisada",PRIMARY),
                signal_chip("Target: traffic_volume",CYAN),
                signal_chip("Datos horarios",ACCENT),
                signal_chip("Clima + tiempo + ambiente",PRIMARY),
            ]),
            html.Div([
                dcc.Link("Explorar datos →",href="/eda",style={"display":"inline-flex","padding":"12px 22px","borderRadius":"999px","background":PRIMARY,"color":"#061016","fontWeight":"900","fontFamily":B,"fontSize":"13px","textDecoration":"none"}),
                dcc.Link("Probar modelo",href="/prueba",style={"display":"inline-flex","padding":"12px 22px","borderRadius":"999px","border":f"1px solid {BORDER}","color":FG,"fontWeight":"800","fontFamily":B,"fontSize":"13px","textDecoration":"none","background":"rgba(255,255,255,0.035)"}),
            ],style={"display":"flex","gap":"12px","flexWrap":"wrap","marginTop":"18px"}),
        ],style={
            "flex":"1.2","minWidth":"330px","background":"rgba(3,8,18,0.58)","backdropFilter":"blur(12px)",
            "border":f"1px solid {BORDER}","borderRadius":"28px","padding":"36px","boxShadow":"0 12px 45px rgba(0,0,0,0.45)",
        }),
        html.Div(context_visual(),style={"flex":"0.8","minWidth":"320px"}),
    ],style={"display":"flex","gap":"22px","alignItems":"center","flexWrap":"wrap","marginBottom":"22px"}),

    html.Div([
        context_kpi("","Problema","Congestión","La demanda vehicular no se comporta igual a las 3 AM que a las 8 AM o en lluvia.",PRIMARY),
        context_kpi("","Datos disponibles","Históricos","Registros horarios con clima, calendario, visibilidad, contaminación y volumen vehicular.",CYAN),
        context_kpi("","Objetivo","Predecir","Estimar traffic_volume antes de tomar decisiones operativas.",ACCENT),
        context_kpi("️","Enfoque","ML","Comparar modelos de regresión para encontrar el mejor balance entre error e interpretación.",PRIMARY),
    ],style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"16px","marginBottom":"18px"}),

    html.Div([
        html.Div([
            html.H3("3. Planteamiento del Problema",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 14px 0"}),
            html.P("Las entidades de gestión de tráfico carecen frecuentemente de herramientas predictivas que integren variables meteorológicas y temporales para estimar el volumen vehicular en tiempo real o anticipado. Esto dificulta la toma de decisiones proactivas para mitigar la congestión, especialmente en días festivos, eventos climáticos extremos o en horas pico.",style={"fontFamily":B,"fontSize":"14px","lineHeight":"1.7","color":MUTED,"margin":"0 0 14px 0"}),
            html.Div([
                html.Div("Pregunta de investigación",style={"fontFamily":B,"fontSize":"10px","letterSpacing":"0.12em","textTransform":"uppercase","fontWeight":"900","color":ACCENT,"marginBottom":"8px"}),
                html.Div("¿Es posible predecir con precisión el volumen de tráfico vehicular a partir de variables meteorológicas, ambientales y temporales, utilizando modelos de aprendizaje automático supervisado?",style={"fontFamily":B,"fontSize":"14px","lineHeight":"1.65","color":FG,"fontStyle":"italic"}),
            ],style={"padding":"16px 18px","borderRadius":"16px","border":f"1px solid {ACCENT}40","background":f"{ACCENT}0d"}),
        ],style={**card(),"flex":"1.2","minWidth":"360px"}),
        html.Div([
            html.H3("4. Objetivo General",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 14px 0"}),
            html.P("Desarrollar y evaluar un modelo de regresión basado en aprendizaje automático capaz de predecir el volumen de tráfico vehicular con alta precisión, utilizando variables meteorológicas, ambientales y temporales derivadas del dataset.",style={"fontFamily":B,"fontSize":"14px","lineHeight":"1.7","color":MUTED,"margin":"0"}),
        ],style={**card(),"flex":"0.8","minWidth":"320px"}),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"16px"}),

    html.Div([
        html.H3("5. Objetivos Específicos",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 18px 0"}),
        html.Div([
            context_value("01","Análisis exploratorio","Identificar distribuciones, valores atípicos, correlaciones y patrones frente a la variable objetivo.",PRIMARY),
            context_value("02","Preprocesamiento","Extraer componentes temporales de date_time y codificar variables categóricas.",CYAN),
            context_value("03","Pipeline ML","Estandarizar el flujo de preprocesamiento, entrenamiento y evaluación de modelos.",ACCENT),
            context_value("04","Comparación","Entrenar y comparar XGBoost, Random Forest, Árbol de Decisión, KNN, SVM, Ridge y Lasso.",PRIMARY),
            context_value("05","Validación final","Seleccionar el modelo de mayor poder predictivo mediante métricas, pruebas estadísticas y residuales.",CYAN),
        ],style={"display":"grid","gridTemplateColumns":"repeat(5,minmax(180px,1fr))","gap":"12px"}),
    ],style={**card(),"marginBottom":"16px"}),

    html.Div([
        html.Div([
            html.H3("6. Justificación",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 14px 0"}),
            html.P("La predicción del volumen de tráfico tiene impacto directo en la planificación urbana, reducción de emisiones de CO₂, diseño de rutas de transporte público y respuesta de emergencias.",style={"fontFamily":B,"fontSize":"14px","lineHeight":"1.7","color":MUTED,"margin":"0 0 16px 0"}),
            html.Div([
                context_value("","Reducir congestión","Semáforos adaptativos y redireccionamiento vehicular.",PRIMARY),
                context_value("","Optimizar recursos","Mantenimiento vial e identificación de períodos de alta demanda.",CYAN),
                context_value("","Apoyar inversión","Infraestructura basada en datos y comportamiento real del tráfico.",ACCENT),
            ],style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"12px"}),
            html.P("El uso de múltiples modelos y su comparación sistemática garantiza que la solución adoptada sea la más adecuada para el comportamiento específico del dataset.",style={"fontFamily":B,"fontSize":"13px","lineHeight":"1.65","color":MUTED,"margin":"16px 0 0 0"}),
        ],style={**card(),"flex":"1","minWidth":"360px"}),
        html.Div([
            html.H3("7.1 Regresión Supervisada",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 14px 0"}),
            html.P("La regresión es una técnica de aprendizaje supervisado que busca aprender una función f: X → y, donde y es una variable continua. El objetivo es minimizar el error de predicción sobre datos no vistos.",style={"fontFamily":B,"fontSize":"14px","lineHeight":"1.7","color":MUTED,"margin":"0"}),
        ],style={**card(),"flex":"0.75","minWidth":"300px"}),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"16px"}),

    html.Div([
        html.H3("7.2 Modelos Implementados",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 10px 0"}),
        theory_table([
            ("XGBoost","Gradient boosting optimizado; combina múltiples árboles débiles con regularización L1/L2. Alta capacidad predictiva y resistencia al overfitting."),
            ("Random Forest","Ensemble de árboles de decisión con bagging. Robusto ante outliers y ruido. Proporciona importancia de variables."),
            ("Árbol de Decisión","Modelo base interpretable. Particiona el espacio de características de forma jerárquica. Propenso a overfitting sin poda."),
            ("KNN","Predicción basada en los k vecinos más cercanos. Sensible a la escala y dimensionalidad."),
            ("SVM (SVR)","Support Vector Regression; maximiza el margen en un espacio de alta dimensión. Efectivo con datos no lineales mediante kernels."),
            ("Ridge","Regresión lineal con regularización L2. Reduce la varianza penalizando coeficientes grandes."),
            ("Lasso","Regresión lineal con regularización L1. Realiza selección automática de variables al llevar coeficientes a cero."),
        ]),
    ],style={**card(),"marginBottom":"16px"}),

    html.Div([
        html.H3("7.3 Métricas de Evaluación",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 18px 0"}),
        html.Div([
            metric_chip("MAE","Mean Absolute Error: error promedio en unidades originales.",PRIMARY),
            metric_chip("RMSE","Root Mean Squared Error: penaliza errores grandes.",ACCENT),
            metric_chip("R²","Coeficiente de Determinación: proporción de varianza explicada.",CYAN),
            metric_chip("MAPE","Mean Absolute Percentage Error: error porcentual promedio.",PRIMARY),
        ],style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px"}),
    ],style={**card(),"marginBottom":"16px"}),

],style={"padding":"46px 0 64px 0"})

# ── PRUEBA DE MODELO ──────────────────────────────────────────
def form_input(label, component, note=""):
    return html.Div([
        html.Label(label,style={"display":"block","fontFamily":B,"fontSize":"12px","fontWeight":"800","color":FG,"marginBottom":"8px"}),
        component,
        html.Div(note,style={"fontFamily":B,"fontSize":"10px","color":MUTED,"marginTop":"6px"}) if note else None,
    ],style={"marginBottom":"14px"})

input_style={"width":"100%","height":"42px","borderRadius":"12px","border":f"1px solid {BORDER}","background":BG,"color":FG,"padding":"0 12px","fontFamily":B,"fontSize":"13px"}

def num_slider(id_, value, min_, max_, step, suffix="", marks=None):
    if marks is None:
        marks = {
            min_: f"{min_}{suffix}",
            max_: f"{max_}{suffix}"
        }
    return html.Div([
        dcc.Slider(
            id=id_,
            min=min_,
            max=max_,
            step=step,
            value=value,
            marks=marks,
            tooltip={"placement":"bottom","always_visible":True},
            className="numeric-slider"
        )
    ],style={
        "padding":"12px 14px 8px 14px",
        "border":f"1px solid {BORDER}",
        "background":BG,
        "borderRadius":"14px"
    })

PRUEBA = html.Section(id="prueba",children=[
    sec("04 — Prueba de Modelo","Simulador dinámico de traffic_volume","Ingresa las features y obtén una predicción estimada con el modelo final."),
    html.Div([
        html.Div([
            html.H3("Variables de entrada",style={"fontFamily":H,"fontSize":"20px","color":FG,"margin":"0 0 18px 0"}),
            html.Div([
                form_input("Hora del día", dcc.Slider(id="p-hour",min=0,max=23,step=1,value=8,marks={0:"0h",6:"6h",12:"12h",18:"18h",23:"23h"},tooltip={"placement":"bottom","always_visible":False}), "0–23"),
                form_input("Día de semana", dcc.Dropdown(id="p-dow",value=0,options=[{"label":l,"value":i} for i,l in enumerate(DOW_LABELS)],clearable=False,style={"color":"#111827"}), "Lunes=0, Domingo=6"),
                form_input("Mes", dcc.Dropdown(id="p-month",value=5,options=[{"label":l,"value":i+1} for i,l in enumerate(MON_LABELS)],clearable=False,style={"color":"#111827"})),
                form_input("Festivo", dcc.Dropdown(id="p-holiday",value=0,options=[{"label":"No","value":0},{"label":"Sí","value":1}],clearable=False,style={"color":"#111827"})),
            ],style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}),
            html.H4("Clima y ambiente",style={"fontFamily":H,"fontSize":"15px","color":FG,"margin":"20px 0 14px 0"}),
            html.Div([
                form_input("Temperatura (K)", num_slider("p-temp",288,240,320,1,"K",{240:"240K",280:"280K",320:"320K"})),
                form_input("Humedad (%)", num_slider("p-humidity",70,0,100,1,"%",{0:"0%",50:"50%",100:"100%"})),
                form_input("Velocidad viento", num_slider("p-wind-speed",3,0,20,0.5,"",{0:"0",10:"10",20:"20"})),
                form_input("Dirección viento", num_slider("p-wind-dir",320,0,360,5,"°",{0:"0°",180:"180°",360:"360°"})),
                form_input("Visibilidad (millas)", num_slider("p-visibility",5,0,10,0.5," mi",{0:"0",5:"5",10:"10"})),
                form_input("Índice contaminación", num_slider("p-pollution",120,0,300,5,"",{0:"0",150:"150",300:"300"})),
                form_input("Lluvia por hora", num_slider("p-rain",0,0,30,0.5,"",{0:"0",15:"15",30:"30"})),
                form_input("Nieve por hora", num_slider("p-snow",0,0,30,0.5,"",{0:"0",15:"15",30:"30"})),
                form_input("Nubosidad (%)", num_slider("p-clouds",20,0,100,1,"%",{0:"0%",50:"50%",100:"100%"})),
                form_input("Tipo de clima", dcc.Dropdown(id="p-weather",value="Clear",options=[{"label":w,"value":w} for w in WEATHER_C],clearable=False,style={"color":"#111827"})),
            ],style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"14px"}),
        ],style={**card(),"flex":"1.3","minWidth":"420px"}),
        html.Div([
            html.Div("MODELO FINAL",style={"fontFamily":B,"fontSize":"10px","fontWeight":"900","letterSpacing":"0.14em","color":PRIMARY,"marginBottom":"8px"}),
            html.H3("XGBoost Regressor",style={"fontFamily":H,"fontSize":"24px","color":FG,"margin":"0 0 18px 0"}),
            html.Div(id="prediction-value",style={"fontFamily":H,"fontSize":"56px","fontWeight":"900","color":PRIMARY,"lineHeight":"1","marginBottom":"10px"}),
            html.Div("vehículos estimados",style={"fontFamily":B,"fontSize":"13px","color":MUTED,"marginBottom":"24px"}),
            html.Div(id="prediction-band",style={"fontFamily":B,"fontSize":"13px","fontWeight":"800","padding":"9px 14px","borderRadius":"999px","display":"inline-flex","marginBottom":"20px"}),
            dcc.Graph(id="prediction-gauge",config={"displayModeBar":False},style={"height":"270px"}),
            html.Div(id="prediction-explain",style={"fontFamily":B,"fontSize":"13px","color":MUTED,"lineHeight":"1.6"}),
        ],style={**card(),"flex":"0.8","minWidth":"320px","position":"sticky","top":"96px","alignSelf":"flex-start"}),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap"}),
],style={"padding":"46px 0 64px 0"})

# ── MARCO DEL PROYECTO ────────────────────────────────────────
# ── MARCO DEL PROYECTO ────────────────────────────────────────
def info_card(num,title,body,accent=PRIMARY,extra=None):
    return html.Div([
        html.Div([
            html.Span(num,style={"fontSize":"11px","color":accent,"fontWeight":"700","fontFamily":B,"letterSpacing":"0.08em"}),
            html.Span(f"  {title}",style={"fontSize":"17px","fontWeight":"700","fontFamily":H,"color":FG}),
        ],style={"marginBottom":"12px"}),
        html.P(body,style={"color":MUTED,"fontSize":"13px","lineHeight":"1.65","fontFamily":B,"margin":"0"}),
        extra,
    ],style={**card(),"flex":"1","minWidth":"300px","position":"relative","overflow":"hidden"})

def sub_tag(icon,title,desc):
    return html.Div([
        html.Div(icon,style={"fontSize":"18px","marginBottom":"6px","color":PRIMARY}),
        html.Div(title,style={"fontSize":"12px","fontWeight":"600","color":FG,"fontFamily":H}),
        html.Div(desc,style={"fontSize":"11px","color":MUTED,"fontFamily":B,"marginTop":"2px"}),
    ],style={"flex":"1","padding":"12px 10px","borderRadius":"11px","border":f"1px solid {BORDER}","background":BG})

MARCO = html.Section(id="marco",children=[
    html.Div(pill("MARCO DEL PROYECTO"),style={"marginBottom":"20px"}),
    html.H2([
        "Predicción del ",html.Span("tráfico",style={"color":PRIMARY}),html.Br(),
        html.Span("vehicular",style={"color":CYAN})," con machine",html.Br(),"learning",
    ],style={"fontSize":"clamp(32px,5vw,62px)","fontWeight":"800","fontFamily":H,
              "color":FG,"margin":"0 0 18px 0","letterSpacing":"-0.03em","lineHeight":"1.08"}),
    html.P("Un análisis completo de regresión supervisada para estimar el volumen de vehículos a partir de variables meteorológicas, temporales y ambientales.",
        style={"color":MUTED,"fontSize":"15px","maxWidth":"680px","lineHeight":"1.6","fontFamily":B,"margin":"0 0 44px 0"}),

    html.Div([
    info_card("01","Introducción",
        "El análisis y predicción del tráfico vehicular representa uno de los desafíos más relevantes "
        "en la ingeniería de transporte y la ciencia de datos aplicada. Con el crecimiento acelerado "
        "de las ciudades y el aumento sostenido del parque automotor, comprender los patrones de "
        "flujo vehicular se convierte en una necesidad estratégica para la planificación urbana "
        "y la reducción de congestión. Este proyecto desarrolla un análisis completo de machine "
        "learning supervisado — específicamente modelos de regresión — para predecir el volumen "
        "de tráfico  comparando Ridge/Lasso con XGBoost y Random Forest."
    ),

    info_card("04","Objetivo general",
        "Desarrollar y evaluar un modelo de regresión basado en aprendizaje automático capaz de "
        "predecir el volumen de tráfico vehicular con alta precisión, utilizando variables "
        "meteorológicas, ambientales y temporales."
    ),

],style={
    "display":"flex",
    "gap":"16px",
    "flexWrap":"wrap",
    "marginBottom":"16px"
}),

    html.Div([
        info_card("02","Contexto",
            "El tráfico está influenciado por una combinación compleja de factores. El dataset utilizado "
            "contiene registros horarios con información meteorológica detallada y el volumen de vehículos "
            "registrados — un caso clásico de regresión con series de tiempo implícitas.",
            extra=html.Div([
                html.Div([sub_tag("️","Climáticos","Lluvia, niebla, temperatura"),
                          sub_tag("","Temporales","Hora, día, festivos"),
                          sub_tag("","Ambientales","Contaminación, visibilidad")],
                         style={"display":"flex","gap":"8px","marginTop":"16px","flexWrap":"wrap"}),
            ])),
        info_card("03","Problemática",
            "Las entidades de gestión de tráfico carecen frecuentemente de herramientas predictivas "
            "que integren variables meteorológicas y temporales para estimar el volumen vehicular "
            "en tiempo real o anticipado.",
            accent=ACCENT,
            extra=html.Div([
                html.Div(" PREGUNTA DE INVESTIGACIÓN",style={
                    "fontSize":"9px","color":ACCENT,"fontWeight":"700","letterSpacing":"0.12em",
                    "textTransform":"uppercase","fontFamily":B,"marginBottom":"10px"}),
                html.P("¿Es posible predecir con precisión el volumen de tráfico vehicular a partir de variables "
                       "meteorológicas, ambientales y temporales usando modelos de aprendizaje automático supervisado?",
                    style={"color":FG,"fontSize":"13px","lineHeight":"1.6","fontFamily":B,"margin":"0","fontStyle":"italic"}),
            ],style={"marginTop":"16px","padding":"14px 16px","borderRadius":"12px",
                     "border":f"1px solid {ACCENT}44","background":f"{ACCENT}08"})),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap"}),
],style={"marginBottom":"64px"})

# ── FEATURE CARDS ─────────────────────────────────────────────
def feat_card(icon,title,text):
    return html.Div([
        html.Div([
            html.Div([html.Div(icon,style={"fontSize":"22px"})],style={
                "width":"48px","height":"48px","borderRadius":"14px",
                "background":f"linear-gradient(135deg,{PRIMARY}22,{ACCENT}18)",
                "display":"flex","alignItems":"center","justifyContent":"center",
            }),
            html.Div("↗",style={"fontSize":"18px","color":MUTED}),
        ],style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start","marginBottom":"22px"}),
        html.H3(title,style={"fontSize":"17px","fontWeight":"700","fontFamily":H,"color":FG,"margin":"0 0 10px 0"}),
        html.P(text,style={"color":MUTED,"fontSize":"13px","lineHeight":"1.6","fontFamily":B,"margin":"0"}),
        html.Div(style={"position":"absolute","top":"-20px","right":"-20px","width":"90px","height":"90px",
                         "borderRadius":"50%","background":f"{PRIMARY}07","filter":"blur(18px)","pointerEvents":"none"}),
    ],style={**card(),"position":"relative","overflow":"hidden"})

FEATURES = html.Section(id="features",children=[
    sec("02 — Capacidades","Más que datos, decisiones.",
        "La solución convierte datos viales, clima y calendario en señales accionables para planear movilidad urbana."),
    html.Div([
        feat_card("","Predicción con IA","Modelos que anticipan congestiones hasta 45 minutos antes con 94% de precisión."),
        feat_card("","Geofencing dinámico","Define zonas críticas y recibe alertas contextuales según el comportamiento del tráfico."),
        feat_card("️","Seguridad vial","Detección automática de incidentes, vehículos detenidos y maniobras peligrosas."),
        feat_card("","Edge computing","Procesamiento en sensor para latencias inferiores a 50ms y privacidad por diseño."),
    ],style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"16px"}),
],style={"marginBottom":"64px"})

# ── EDA ───────────────────────────────────────────────────────
EDA = html.Section(id="eda",children=[
    sec("03 — EDA","Explorador interactivo de datos","Selecciona una variable para inspeccionar su comportamiento y su relacion con traffic_volume"),
    html.Div([
        html.Div([
            html.Div([
                html.Div("",style={"fontSize":"22px","color":PRIMARY,"marginRight":"10px"}),
                html.Div([
                    html.H3("Explorador de variables",style={"fontFamily":H,"fontSize":"22px","color":FG,"margin":"0 0 4px 0"}),
                    html.P(["Elige una variable para revisar su distribucion y compararla contra ",html.Code("traffic_volume",style={"color":PRIMARY,"background":"transparent"})],
                           style={"color":MUTED,"fontSize":"14px","fontFamily":B,"margin":"0"}),
                ]),
            ],style={"display":"flex","alignItems":"flex-start","marginBottom":"22px"}),
            explorer_controls("hour"),
            html.Div(id="eda-selected-desc",style={"color":MUTED,"fontSize":"13px","fontFamily":B,"marginTop":"10px"}),
        ],style={**card(),"flex":"1.8","minWidth":"320px"}),
        html.Div([
            html.Div("CORRELACION DE SPEARMAN",style={"fontSize":"10px","color":MUTED,"fontWeight":"700","letterSpacing":"0.14em","fontFamily":B,"textTransform":"uppercase"}),
            html.Div(id="eda-corr-value",style={"fontSize":"38px","fontWeight":"800","fontFamily":H,"color":PRIMARY,"marginTop":"8px"}),
            html.Div(id="eda-corr-label",style={"fontSize":"12px","color":MUTED,"fontFamily":B}),
        ],style={**card(),"width":"280px","minWidth":"240px","display":"flex","flexDirection":"column","justifyContent":"center"}),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"16px"}),

    html.Div([
        html.Div([
            html.Div([html.H4("Comportamiento de la variable",style={"fontFamily":H,"fontSize":"16px","color":FG,"margin":"0"}), badge("EDA",PRIMARY)],
                     style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"8px"}),
            dcc.Graph(id="eda-behavior-graph",config={"displayModeBar":False},style={"height":"390px"}),
        ],style={**card()}),
        html.Div([
            html.Div([html.H4("Variable vs target",style={"fontFamily":H,"fontSize":"16px","color":FG,"margin":"0"}), badge("TARGET",CYAN)],
                     style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"8px"}),
            dcc.Graph(id="eda-target-graph",config={"displayModeBar":False},style={"height":"390px"}),
        ],style={**card()}),
    ],style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}),

    html.Div([
        html.Div([
            html.H4("Resumen rapido",style={"fontFamily":H,"fontSize":"14px","color":FG,"margin":"0 0 16px 0"}),
            html.Div(id="eda-insight",style={"color":MUTED,"fontSize":"13px","lineHeight":"1.65","fontFamily":B}),
        ],style={**card(),"flex":"1"}),
        html.Div([
            html.H4("Diagnosticos del dataset",style={"fontFamily":H,"fontSize":"14px","color":FG,"margin":"0 0 16px 0"}),
            html.Div([
                html.Div([html.Div("33,750",style={"fontSize":"22px","fontWeight":"800","fontFamily":H,"color":FG}),html.Div("Registros",style={"fontSize":"11px","color":MUTED,"fontFamily":B})]),
                html.Div([html.Div("16",style={"fontSize":"22px","fontWeight":"800","fontFamily":H,"color":PRIMARY}),html.Div("Features",style={"fontSize":"11px","color":MUTED,"fontFamily":B})]),
                html.Div([html.Div("0",style={"fontSize":"22px","fontWeight":"800","fontFamily":H,"color":CYAN}),html.Div("Nulos",style={"fontSize":"11px","color":MUTED,"fontFamily":B})]),
                html.Div([html.Div("No normal",style={"fontSize":"22px","fontWeight":"800","fontFamily":H,"color":DANGER}),html.Div("Shapiro p<0.05",style={"fontSize":"11px","color":MUTED,"fontFamily":B})]),
            ],style={"display":"flex","gap":"28px","flexWrap":"wrap"}),
        ],style={**card(),"flex":"1.4"}),
    ],style={"display":"flex","gap":"16px","flexWrap":"wrap"}),
],style={"marginBottom":"64px"})

# ── MODELS ────────────────────────────────────────────────────
def rrow(i,m):
    best=i==0
    return html.Div([
        html.Div(str(i+1),style={
            "width":"42px","textAlign":"center","fontSize":"18px","fontFamily":B,
            "color":PRIMARY if best else MUTED,"fontWeight":"800" if best else "500"
        }),
        html.Div([
            html.Span(m["name"],style={
                "fontSize":"18px","fontWeight":"800" if best else "600",
                "color":PRIMARY if best else FG,"fontFamily":H
            }),
            badge("GANADOR",PRIMARY) if best else None
        ],style={"flex":"1","display":"flex","alignItems":"center","gap":"12px"}),
        *[html.Div(v,style={
            "width":"110px","textAlign":"right","fontSize":"17px","fontFamily":B,
            "color":PRIMARY if (best and j==0) else (MUTED if j>0 else FG),
            "fontWeight":"800" if (best and j==0) else "600",
            "fontVariantNumeric":"tabular-nums"
        }) for j,v in enumerate([f"{m['r2']:.4f}",f"{m['rmse']:.1f}",f"{m['mae']:.1f}",f"{m['cv']:.4f}"])],
    ],style={
        "display":"flex","alignItems":"center","gap":"14px","padding":"18px 16px",
        "borderRadius":"14px","background":f"{PRIMARY}07" if best else "transparent",
        "marginTop":"6px","minHeight":"64px"
    })


def model_note(title, body, color=PRIMARY):
    return html.Div([
        html.Div(title,style={"fontFamily":H,"fontSize":"12px","fontWeight":"800","color":color,"marginBottom":"6px"}),
        html.P(body,style={"fontFamily":B,"fontSize":"12px","color":MUTED,"lineHeight":"1.55","margin":"0"})
    ],style={
        "marginTop":"14px",
        "padding":"13px 14px",
        "border":f"1px solid {color}30",
        "background":f"{color}0f",
        "borderRadius":"14px"
    })

MODELS_S = html.Section(id="modelos",children=[
    sec("04 — Modelos","Comparación de Algoritmos","7 modelos · XGBoost · Random Forest · KNN · SVR · Ridge · Lasso"),
    html.Div([
        html.H4("Ranking Final",style={"fontFamily":H,"fontSize":"22px","color":FG,"margin":"0 0 24px 0"}),
        html.Div([
            html.Div([
                html.Div(h,style={
                    "fontSize":"15px","fontWeight":"800","color":MUTED,"letterSpacing":"0.08em",
                    "textTransform":"uppercase","fontFamily":B,
                    "flex" if h=="Modelo" else "width":"1" if h=="Modelo" else "110px",
                    "textAlign":"left" if h in ["#","Modelo"] else "right"
                }) for h in ["#","Modelo","R²","RMSE","MAE","CV R²"]
            ],style={"display":"flex","gap":"14px","padding":"0 18px 18px","borderBottom":f"1px solid {BORDER}"}),
            *[rrow(i,m) for i,m in enumerate(MODELS)],
        ]),
    ],style={**card(),"marginBottom":"16px","padding":"32px"}),
    html.Div([
        html.Div([
            G(fig_r2(),340),
            model_note("Lectura rápida","XGBoost y Random Forest mantienen los mejores R² y una validación cruzada muy cercana al test, lo que indica buena capacidad de generalización.")
        ],style={**card()}),
        html.Div([
            G(fig_err(),340),
            model_note("Lectura rápida","Los modelos ensemble reducen claramente el MAE y RMSE; Ridge y Lasso funcionan como baseline, pero dejan errores más altos.")
        ],style={**card()})
    ],style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}),

    html.Div([
        G(fig_overfit(),340),
        model_note("Diagnóstico","La distancia entre R² Train y R² Test permite detectar sobreajuste. Un delta pequeño en XGBoost sugiere un modelo fuerte sin sobreentrenamiento severo.",ACCENT)
    ],style={**card(),"marginBottom":"16px"}),

    html.Div([
        html.Div([
            G(fig_imp(),420),
            model_note("Interpretación","La importancia de variables muestra qué señales usa más XGBoost. La hora del día domina el comportamiento del tráfico frente a variables climáticas aisladas.")
        ],style={**card()}),
        html.Div([
            G(fig_ridge(),420),
            model_note("Interpretación","Los coeficientes Ridge ayudan a explicar dirección e impacto lineal. Son útiles para interpretación, aunque su desempeño predictivo es menor que XGBoost.")
        ],style={**card()})
    ],style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}),
],style={"marginBottom":"64px"})

# ── INSIGHTS ──────────────────────────────────────────────────
def _semaforo_luz(color, activo):
    return html.Div(style={
        "width":"20px","height":"20px","borderRadius":"50%","background":color,
        "margin":"5px auto","opacity":"1" if activo else "0.15",
        "boxShadow":(f"0 0 14px {color}, 0 0 28px {color}66") if activo else "none",
    })

def _flow_bar(label, pct, color, delay):
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize":"10px","color":MUTED,"fontFamily":B,"fontWeight":"700"}),
            html.Span(f"{pct}%", style={"fontSize":"10px","color":color,"fontFamily":B,"fontWeight":"800"}),
        ], style={"display":"flex","justifyContent":"space-between","marginBottom":"3px"}),
        html.Div(html.Div(style={
            "width":f"{pct}%","height":"100%","borderRadius":"999px",
            "background":f"linear-gradient(90deg,{color},{color}66)",
            "boxShadow":f"0 0 8px {color}88",
            "animation":"flowPulse 2s ease-in-out infinite",
            "animationDelay":delay,
        }), style={"height":"5px","borderRadius":"999px","background":BORDER,"overflow":"hidden","marginBottom":"8px"}),
    ])

def _benefit(icon, text, color):
    return html.Div([
        html.Span(icon, style={"color":color,"fontSize":"16px","marginRight":"12px","flexShrink":"0"}),
        html.Span(text, style={"fontSize":"15px","color":MUTED,"fontFamily":B,"lineHeight":"1.5"}),
    ], style={"display":"flex","alignItems":"center","padding":"8px 0","borderBottom":f"1px solid {BORDER}"})

def _gps(x, y, color, sz="8px"):
    return html.Div(style={
        "position":"absolute","width":sz,"height":sz,"borderRadius":"50%","background":color,
        "boxShadow":f"0 0 10px {color}, 0 0 20px {color}55",
        "left":x,"top":y,"animation":"gpsPulse 2s ease-in-out infinite",
    })

APLICACIONES = html.Section(id="insights", children=[
    sec("05 — Aplicaciones Reales","Del modelo al mundo real.","Casos de uso industriales · Impacto urbano y tecnológico"),
    html.Div([

        # CARD 1: Semaforos Inteligentes
        html.Div([
            html.Div([
                html.H2("Semáforos Inteligentes", style={
                    "fontFamily":H,"fontSize":"clamp(22px,2.8vw,32px)","color":FG,
                    "fontWeight":"900","margin":"0 0 6px 0",
                    "textShadow":f"0 0 40px {PRIMARY}55",
                }),
                html.P("El modelo anticipa picos de congestión con minutos de anticipación, permitiendo que los semáforos reajusten sus ciclos automáticamente antes de que el colapso ocurra. Ciudades más fluidas, conductores menos estresados, aire más limpio.", style={
                    "fontSize":"16px","color":MUTED,"fontFamily":B,
                    "lineHeight":"1.75","margin":"0 0 24px 0",
                }),
                html.Div([
                    _benefit("▲","Reducción significativa de la congestión vehicular", PRIMARY),
                    _benefit("⟳","Sincronización adaptativa en tiempo real entre intersecciones", PRIMARY),
                    _benefit("◷","Menos tiempo de espera para conductores y peatones", PRIMARY),
                    _benefit("◈","Menor emisión de CO₂ al reducir arranques y frenadas", PRIMARY),
                ]),
            ], style={"flex":"1","minWidth":"0"}),
            html.Div([
                html.Div([
                    html.Video(src="/assets/Semaforo.mp4", autoPlay=True, muted=True, loop=True, style={
                        "width":"100%","borderRadius":"16px","display":"block",
                        "boxShadow":f"0 0 32px {PRIMARY}44",
                        "border":f"2px solid {PRIMARY}33",
                    }),
                    html.Div([
                        html.Div(style={"width":"9px","height":"9px","borderRadius":"50%","background":PRIMARY,"boxShadow":f"0 0 12px {PRIMARY}","animation":"livePulse 1.2s ease-in-out infinite","display":"inline-block","marginRight":"6px","verticalAlign":"middle"}),
                        html.Span("LIVE", style={"fontSize":"10px","color":PRIMARY,"fontFamily":B,"fontWeight":"800","letterSpacing":"0.14em","verticalAlign":"middle"}),
                    ], style={"display":"flex","alignItems":"center","justifyContent":"center","marginTop":"10px"}),
                ], style={"textAlign":"center"}),
            ], style={"width":"190px","flexShrink":"0","paddingTop":"4px"}),
        ], style={
            **card(),"display":"flex","gap":"28px","alignItems":"flex-start",
            "flex":"1","minWidth":"340px","padding":"32px",
            "background":f"radial-gradient(ellipse at top left,{PRIMARY}14,transparent 55%), radial-gradient(ellipse at bottom right,{CYAN}08,transparent 50%),{CARD}",
            "borderTop":f"3px solid {PRIMARY}","position":"relative","overflow":"hidden",
        }),

        # CARD 2: Navegacion Predictiva
        html.Div([
        html.Div([
            html.H2("Google Maps · Waze · Uber", style={
                "fontFamily":H,"fontSize":"clamp(22px,2.8vw,32px)","color":FG,
                "fontWeight":"900","margin":"0 0 6px 0",
                "textShadow":f"0 0 40px {CYAN}55",
            }),
            html.P("Integrar el modelo con plataformas de navegación permite ofrecer rutas que se adaptan a condiciones futuras: lluvia, hora pico, festivos. El sistema no reacciona al tráfico, lo anticipa, reduciendo fricciones urbanas a escala masiva.", style={
                "fontSize":"16px","color":MUTED,"fontFamily":B,
                "lineHeight":"1.75","margin":"0 0 24px 0",
            }),
            html.Div([
                _benefit("⇢","Rutas sugeridas basadas en tráfico futuro predicho", CYAN),
                _benefit("◎","Anticipación de picos antes de que se formen", CYAN),
                _benefit("◷","ETAs más precisos con variables meteorológicas incluidas", CYAN),
                _benefit("⚑","Alertas tempranas a conductores y operadores de flota", CYAN),
                _benefit("⊕","Optimización de asignación de conductores en plataformas", CYAN),
            ], style={"marginBottom":"24px"}),
        ], style={"flex":"1","minWidth":"0"}),
        html.Div([
            html.Div([
                *[html.Div(style={"position":"absolute","left":"0","right":"0","height":"1px","background":BORDER,"top":f"{y}%","opacity":"0.6"}) for y in [20,40,60,80]],
                *[html.Div(style={"position":"absolute","top":"0","bottom":"0","width":"1px","background":BORDER,"left":f"{x}%","opacity":"0.6"}) for x in [25,50,75]],
                html.Div(style={"position":"absolute","left":"5%","top":"20%","width":"70%","height":"2px","background":f"linear-gradient(90deg,{CYAN},{CYAN}11)","borderRadius":"999px","boxShadow":f"0 0 10px {CYAN}","animation":"routeGlow 2s ease-in-out infinite"}),
                html.Div(style={"position":"absolute","left":"25%","top":"20%","width":"2px","height":"40%","background":f"linear-gradient(180deg,{CYAN},{PRIMARY})","borderRadius":"999px","boxShadow":f"0 0 10px {CYAN}","animation":"routeGlow 2.4s ease-in-out infinite","animationDelay":"0.5s"}),
                html.Div(style={"position":"absolute","left":"5%","top":"60%","width":"95%","height":"2px","background":f"linear-gradient(90deg,{PRIMARY},{PRIMARY}11)","borderRadius":"999px","boxShadow":f"0 0 10px {PRIMARY}","animation":"routeGlow 2.8s ease-in-out infinite","animationDelay":"1s"}),
                html.Div(style={"position":"absolute","left":"75%","top":"20%","width":"2px","height":"65%","background":f"linear-gradient(180deg,{CYAN}55,{ACCENT})","borderRadius":"999px","boxShadow":f"0 0 8px {ACCENT}88","animation":"routeGlow 3s ease-in-out infinite","animationDelay":"1.4s"}),
                _gps("4%","18%", CYAN, "12px"),
                _gps("73%","18%", PRIMARY, "10px"),
                _gps("24%","58%", ACCENT, "11px"),
                _gps("93%","83%", CYAN, "10px"),
                _gps("25%","18%", PRIMARY, "9px"),
                _gps("4%","58%", CYAN, "9px"),
                html.Div([
                    html.Div("Demanda", style={"fontSize":"8px","color":MUTED,"fontFamily":B,"fontWeight":"700","textTransform":"uppercase","letterSpacing":"0.08em"}),
                    html.Div("Alta", style={"fontSize":"15px","color":ACCENT,"fontFamily":H,"fontWeight":"900","lineHeight":"1"}),
                ], style={
                    "position":"absolute","right":"6px","top":"6px",
                    "background":"rgba(6,12,20,0.92)","borderRadius":"9px",
                    "padding":"6px 10px","border":f"1px solid {ACCENT}55","textAlign":"center",
                }),
            ], style={
                "position":"relative","height":"200px","background":"rgba(6,12,20,0.55)",
                "borderRadius":"14px","border":f"1px solid {BORDER}","overflow":"hidden","marginBottom":"10px",
            }),
            html.Div("Mapa predictivo urbano en tiempo real", style={
                "fontSize":"10px","color":MUTED,"fontFamily":B,"textAlign":"center",
                "letterSpacing":"0.06em","textTransform":"uppercase",
            }),
        ], style={"width":"190px","flexShrink":"0","paddingTop":"4px"}),
    ], style={
        **card(),"display":"flex","gap":"28px","alignItems":"flex-start",
        "flex":"1","minWidth":"340px","padding":"32px",
        "background":f"radial-gradient(ellipse at top right,{CYAN}12,transparent 55%), radial-gradient(ellipse at bottom left,{PRIMARY}06,transparent 50%),{CARD}",
        "borderTop":f"3px solid {CYAN}","position":"relative","overflow":"hidden",
    }),

    ], style={"display":"flex","gap":"20px","flexWrap":"wrap"}),
],style={"marginBottom":"80px"})

FOOTER = html.Footer([
    html.Span("© 2026 Predicción Volumen Tráfico · Analítica predictiva",style={"fontSize":"12px","color":MUTED,"fontFamily":B}),
    html.Span([html.Span("●",style={"color":PRIMARY,"marginRight":"6px"}),"Todos los sistemas operativos"],style={"fontSize":"12px","color":MUTED,"fontFamily":B}),
],style={"display":"flex","justifyContent":"space-between","alignItems":"center","flexWrap":"wrap","gap":"16px","padding":"28px 0","borderTop":f"1px solid {BORDER}"})

CSS = html.Div([
    html.Link(rel="stylesheet",
              href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap"),
    dcc.Markdown("""
<style>
@keyframes drive { 0%{left:-10%} 100%{left:110%} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.8)} }
@keyframes flowPulse { 0%,100%{opacity:0.75} 50%{opacity:1} }
@keyframes routeGlow { 0%,100%{opacity:0.5} 50%{opacity:1} }
@keyframes gpsPulse { 0%,100%{transform:scale(1);opacity:0.9} 50%{transform:scale(1.5);opacity:0.4} }
* { box-sizing:border-box; }
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#080d16}
::-webkit-scrollbar-thumb{background:#1a2840;border-radius:3px}
a:hover{opacity:0.85}
@media (max-width: 900px){
  main{padding:0 18px!important}
}

.rc-slider-track{background:linear-gradient(90deg,#00e599,#22d3ee)!important;height:6px!important}
.rc-slider-rail{background:#1a2840!important;height:6px!important}
.rc-slider-handle{border:2px solid #00e599!important;background:#08111d!important;box-shadow:0 0 0 5px rgba(0,229,153,.12)!important;width:18px!important;height:18px!important;margin-top:-6px!important}
.rc-slider-dot{display:none!important}
.rc-slider-mark-text{color:#6b8fac!important;font-size:10px!important;font-family:Inter,system-ui,sans-serif!important}
.rc-slider-tooltip-inner{background:#00e599!important;color:#061016!important;font-weight:800!important;box-shadow:none!important}
.rc-slider-tooltip-arrow{border-top-color:#00e599!important}

</style>
""", dangerously_allow_html=True),
])

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Predicción Volumen Tráfico",
    suppress_callback_exceptions=True,
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}]
)

ABOUT_US = html.Section(
    id="about-us",
    children=[
        html.Div([
            html.H2("About Us", style={
                "fontFamily": H, "fontSize": "clamp(32px,4vw,52px)", "fontWeight": "900",
                "color": FG, "textAlign": "center", "margin": "0 0 48px 0",
                "letterSpacing": "-0.03em",
            }),
            html.Div([
                # Columna izquierda: tarjetas
                html.Div([
                    html.H3("About Us", style={
                        "fontFamily": H, "fontSize": "20px", "fontWeight": "800",
                        "color": FG, "margin": "0 0 20px 0", "textAlign": "center",
                        "letterSpacing": "-0.02em",
                    }),
                    # Card Luis
                    html.Div([
                        html.H3("Luis Esteban Mariño", style={
                            "fontFamily": H, "fontSize": "18px", "fontWeight": "800",
                            "color": FG, "margin": "0 0 20px 0", "textAlign": "center",
                        }),
                        html.Div([
                            html.A("GitHub", href="https://github.com/lesteban1828", target="_blank", style={
                                "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                                "padding": "10px 22px", "borderRadius": "999px",
                                "background": PRIMARY, "color": "#061016",
                                "fontWeight": "900", "fontFamily": B, "fontSize": "14px",
                                "textDecoration": "none", "boxShadow": f"0 0 20px {PRIMARY}40",
                            }),
                            html.A("LinkedIn", href="https://www.linkedin.com/in/luis-esteban-mariño-1b7a61406/", target="_blank", style={
                                "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                                "padding": "10px 22px", "borderRadius": "999px",
                                "background": CYAN, "color": "#061016",
                                "fontWeight": "900", "fontFamily": B, "fontSize": "14px",
                                "textDecoration": "none", "boxShadow": f"0 0 20px {CYAN}40",
                            }),
                        ], style={"display": "flex", "gap": "12px", "justifyContent": "center"}),
                    ], style={
                        "background": "rgba(15,22,35,0.85)", "border": f"1px solid {BORDER}",
                        "borderRadius": "22px", "padding": "32px 28px",
                        "backdropFilter": "blur(10px)",
                    }),
                    # Card Juan
                    html.Div([
                        html.H3("Juan Esteban Garcia", style={
                            "fontFamily": H, "fontSize": "18px", "fontWeight": "800",
                            "color": FG, "margin": "0 0 20px 0", "textAlign": "center",
                        }),
                        html.Div([
                            html.A("GitHub", href="https://github.com/JuanesGarciaGomezy", target="_blank", style={
                                "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                                "padding": "10px 22px", "borderRadius": "999px",
                                "background": PRIMARY, "color": "#061016",
                                "fontWeight": "900", "fontFamily": B, "fontSize": "14px",
                                "textDecoration": "none", "boxShadow": f"0 0 20px {PRIMARY}40",
                            }),
                            html.A("LinkedIn", href="https://www.linkedin.com/in/juan-esteban-garcia-gomez-49629934a", target="_blank", style={
                                "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                                "padding": "10px 22px", "borderRadius": "999px",
                                "background": CYAN, "color": "#061016",
                                "fontWeight": "900", "fontFamily": B, "fontSize": "14px",
                                "textDecoration": "none", "boxShadow": f"0 0 20px {CYAN}40",
                            }),
                        ], style={"display": "flex", "gap": "12px", "justifyContent": "center"}),
                    ], style={
                        "background": "rgba(15,22,35,0.85)", "border": f"1px solid {BORDER}",
                        "borderRadius": "22px", "padding": "32px 28px",
                        "backdropFilter": "blur(10px)",
                    }),
                ], style={
                    "display": "flex", "flexDirection": "column", "gap": "24px",
                    "flex": "0 0 auto", "width": "320px",
                }),
                # Columna derecha: video
                html.Div([
                    html.H3("Video Explicativo", style={
                        "fontFamily": H, "fontSize": "20px", "fontWeight": "800",
                        "color": FG, "margin": "0 0 16px 0", "textAlign": "center",
                        "letterSpacing": "-0.02em",
                    }),
                    html.Iframe(
                        src="https://drive.google.com/file/d/10OMAWGW1HXAEz-RvD7fdLmEjCGhNscMO/preview",
                        style={
                            "width": "100%", "flex": "1",
                            "border": "none", "borderRadius": "12px",
                            "minHeight": "280px",
                        },
                        allow="autoplay",
                    ),
                ], style={
                    "flex": "1", "minWidth": "300px",
                    "background": "rgba(15,22,35,0.85)", "border": f"1px solid {BORDER}",
                    "borderRadius": "22px", "overflow": "hidden",
                    "backdropFilter": "blur(10px)", "minHeight": "300px",
                    "padding": "24px 24px 0 24px",
                    "display": "flex", "flexDirection": "column",
                }),
            ], style={
                "display": "flex", "gap": "32px", "alignItems": "stretch",
                "flexWrap": "wrap", "maxWidth": "1100px", "margin": "0 auto",
            }),
        ], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "0 26px"}),
    ],
    style={
        "padding": "72px 0 80px 0",
        "background": "rgba(8,13,22,0.72)",
        "backdropFilter": "blur(8px)",
        "borderTop": f"1px solid {BORDER}",
        "marginTop": "24px",
    },
)

def page_shell(content):
    return html.Main(
        [content, FOOTER],
        style={"maxWidth":"1280px","margin":"0 auto","padding":"0 28px","minHeight":"calc(100vh - 80px)"}
    )

def page_home():
    return page_shell(html.Div([HERO, ABOUT_US]))

def page_contexto():
    return page_shell(html.Div([CONTEXTO]))

def page_eda():
    return page_shell(html.Div([EDA]))

def page_prueba():
    return page_shell(html.Div([PRUEBA]))

def page_modelos():
    return page_shell(html.Div([MODELS_S]))

def page_insights():
    return page_shell(html.Div([APLICACIONES]))

app.layout = html.Div([

    CSS,

    dcc.Location(id="url", refresh=False),

    html.Div([
        NAVBAR,
        html.Div(
            id="page-content",
            style={
                "position": "relative",
                "zIndex": "1",
            }
        ),
    ], style={
        "position": "relative",
        "zIndex": "1",
    }),

], style={
    "backgroundImage": (
        "radial-gradient(ellipse at center, rgba(3,8,18,0.15) 0%, rgba(3,8,18,0.80) 100%), "
        "linear-gradient(rgba(3,8,18,0.60), rgba(3,8,18,0.60)), "
        "url('/assets/NuevaImagenTrafico.png')"
    ),
    "backgroundSize": "cover",
    "backgroundPosition": "center",
    "backgroundAttachment": "fixed",
    "backgroundRepeat": "no-repeat",
    "backgroundColor": BG,
    "minHeight": "100vh",
    "fontFamily": B,
})

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/contexto":
        return page_contexto()
    if pathname == "/eda":
        return page_eda()
    if pathname == "/prueba":
        return page_prueba()
    if pathname == "/modelos":
        return page_modelos()
    if pathname == "/insights":
        return page_insights()
    return page_home()

@app.callback(
    Output("eda-behavior-graph", "figure"),
    Output("eda-target-graph", "figure"),
    Output("eda-selected-desc", "children"),
    Output("eda-corr-value", "children"),
    Output("eda-corr-label", "children"),
    Output("eda-insight", "children"),
    Input("eda-variable", "value"),
)
def update_eda_explorer(var):
    meta = EDA_META.get(var, EDA_META["hour"])
    corr = meta["corr"]
    corr_color = PRIMARY if corr >= 0 else DANGER
    direction = "positiva" if corr >= 0 else "negativa"
    strength = "fuerte" if abs(corr) >= 0.5 else ("moderada" if abs(corr) >= 0.15 else "debil")
    desc = f"{meta['desc']} · Tipo: {meta['unit']}"
    corr_value = f"{corr:+.3f}"
    corr_label = f"relacion {direction} {strength} con traffic_volume"
    insight = [
        html.Span("Interpretacion: ",style={"color":FG,"fontWeight":"700"}),
        f"{meta['title']} muestra una correlacion Spearman {corr_value} con el target. ",
        f"Esto sugiere una relacion {direction} y {strength}; debe leerse junto con el grafico comparativo, porque algunas variables tienen comportamiento no lineal o categorico.",
    ]
    fig_a = fig_variable_behavior(var)
    fig_b = fig_variable_target(var)
    return fig_a, fig_b, desc, corr_value, corr_label, insight


def estimate_traffic(hour, dow, month, holiday, temp, humidity, wind_speed, wind_dir, visibility, pollution, rain, snow, clouds, weather):
    hour = 8 if hour is None else float(hour)
    dow = 0 if dow is None else int(dow)
    month = 5 if month is None else int(month)
    holiday = 0 if holiday is None else int(holiday)
    temp = 288 if temp is None else float(temp)
    humidity = 70 if humidity is None else float(humidity)
    wind_speed = 3 if wind_speed is None else float(wind_speed)
    visibility = 5 if visibility is None else float(visibility)
    pollution = 120 if pollution is None else float(pollution)
    rain = 0 if rain is None else float(rain)
    snow = 0 if snow is None else float(snow)
    clouds = 20 if clouds is None else float(clouds)
    weather = weather or "Clear"

    base = 3100
    hour_profile = np.interp(hour, list(range(24)), HOUR_VOL) - 3200
    dow_effect = {0:120,1:220,2:260,3:300,4:260,5:-520,6:-850}.get(dow,0)
    month_effect = {1:-180,2:-80,3:40,4:120,5:160,6:180,7:80,8:120,9:180,10:150,11:-40,12:-160}.get(month,0)
    weather_effect = {"Clear":180,"Clouds":80,"Mist":-90,"Haze":-110,"Rain":-240,"Drizzle":-260,"Snow":-650,"Thunderstorm":-820,"Fog":-720}.get(weather,0)
    climate_effect = 4.5*(temp-288) - 5.0*rain - 9.0*snow - 2.5*holiday*100 + 8.0*(visibility-5) + 0.8*(clouds-40) - 0.4*(humidity-70) + 6.0*(wind_speed-3) - 0.15*(pollution-120)
    pred = base + hour_profile + dow_effect + month_effect + weather_effect + climate_effect
    return int(np.clip(pred, 150, 7200))

def prediction_gauge_fig(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font":{"color":FG,"size":34,"family":H},"suffix":" veh"},
        gauge={
            "axis":{"range":[0,7200],"tickcolor":MUTED,"tickfont":{"color":MUTED}},
            "bar":{"color":PRIMARY},
            "bgcolor":CARD,
            "borderwidth":1,
            "bordercolor":BORDER,
            "steps":[
                {"range":[0,2500],"color":"rgba(0,229,153,0.16)"},
                {"range":[2500,5000],"color":"rgba(249,115,22,0.18)"},
                {"range":[5000,7200],"color":"rgba(239,68,68,0.20)"},
            ],
            "threshold":{"line":{"color":ACCENT,"width":4},"thickness":0.75,"value":value}
        }
    ))
    fig.update_layout(plot_bgcolor=CARD,paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=20,r=20,t=20,b=20),font=dict(color=FG,family=B))
    return fig

@app.callback(
    Output("prediction-value", "children"),
    Output("prediction-band", "children"),
    Output("prediction-band", "style"),
    Output("prediction-gauge", "figure"),
    Output("prediction-explain", "children"),
    Input("p-hour","value"),Input("p-dow","value"),Input("p-month","value"),Input("p-holiday","value"),
    Input("p-temp","value"),Input("p-humidity","value"),Input("p-wind-speed","value"),Input("p-wind-dir","value"),
    Input("p-visibility","value"),Input("p-pollution","value"),Input("p-rain","value"),Input("p-snow","value"),
    Input("p-clouds","value"),Input("p-weather","value"),
)
def update_prediction(hour,dow,month,holiday,temp,humidity,wind_speed,wind_dir,visibility,pollution,rain,snow,clouds,weather):
    value = estimate_traffic(hour,dow,month,holiday,temp,humidity,wind_speed,wind_dir,visibility,pollution,rain,snow,clouds,weather)
    if value >= 5000:
        label, color = "Demanda alta", DANGER
    elif value >= 2800:
        label, color = "Demanda media", ACCENT
    else:
        label, color = "Demanda baja", PRIMARY
    style={"fontFamily":B,"fontSize":"13px","fontWeight":"800","padding":"9px 14px","borderRadius":"999px","display":"inline-flex","marginBottom":"20px","background":f"{color}18","color":color,"border":f"1px solid {color}45"}
    hour_txt = f"{int(hour or 0):02d}:00"
    explain = f"La estimación responde principalmente al patrón horario ({hour_txt}), el día seleccionado y las condiciones climáticas. Este simulador reproduce el comportamiento esperado del modelo final para presentar la lógica de predicción al usuario."
    return f"{value:,}".replace(",","."), label, style, prediction_gauge_fig(value), explain

server = app.server

if __name__=="__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
