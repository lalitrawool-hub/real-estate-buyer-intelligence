"""Real Estate Buyer Intelligence dashboard for Parcl."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.clustering import cluster_buyers, evaluate_clusters
from src.preprocessing import ROOT, build_buyer_features, load_data, make_preprocessor
from src.profiling import summarize_segments

st.set_page_config(page_title="Parcl | Buyer Intelligence", page_icon="◆", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root{--canvas:#06080D;--rail:#0B0F17;--panel:#111827;--panel-raised:#181C24;--line:rgba(255,255,255,.085);--ink:#F9FAFB;--muted:#9CA3AF;--quiet:#6B7280;--mint:#0AE8B6;--cyan:#06B6D4;--violet:#818CF8;--amber:#F59E0B;--red:#F87171}
.stApp,[data-testid="stAppViewContainer"]{background:var(--canvas)!important;color:var(--ink);font-family:'Inter',sans-serif}.block-container{padding:1.65rem 2.5rem 3rem;max-width:1728px}
[data-testid="stHeader"]{background:rgba(6,8,13,.92);border-bottom:1px solid var(--line)}[data-testid="stToolbar"]{right:1rem}
h1,h2,h3{font-family:'Plus Jakarta Sans',sans-serif;color:var(--ink);letter-spacing:-.035em}.hero{padding:.6rem 0 1.35rem;margin-bottom:1rem;border-bottom:1px solid var(--line)}.hero-top{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.7rem}.eyebrow{color:var(--mint);text-transform:uppercase;letter-spacing:.16em;font-size:.68rem;font-weight:700}.model-chip{display:inline-flex;align-items:center;gap:.5rem;padding:.38rem .65rem;border:1px solid rgba(10,232,182,.24);border-radius:6px;background:rgba(10,232,182,.08);color:#C8FFF0;font-size:.72rem;font-weight:600;letter-spacing:.035em}.model-chip:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 10px rgba(10,232,182,.65)}.hero h1{font-size:clamp(1.8rem,3vw,2.35rem);line-height:1.16;margin:0}.hero p{color:var(--muted);font-size:.94rem;margin:.45rem 0 0;max-width:780px;line-height:1.6}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(17,24,39,.96),rgba(11,15,23,.92));border:1px solid var(--line);border-radius:10px;padding:1.05rem 1.15rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.045);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:rgba(10,232,182,.38);box-shadow:0 8px 24px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.06)}[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:var(--muted)!important;font-size:.76rem;letter-spacing:.025em}[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:var(--ink)!important;font-family:'Inter',sans-serif;font-variant-numeric:tabular-nums;font-weight:600}
[data-testid="stSidebar"],[data-testid="stSidebar"] section{background:var(--rail)!important;color:var(--ink)!important;border-right:1px solid var(--line)}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{color:var(--ink)!important;font-family:'Plus Jakarta Sans',sans-serif}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1{font-size:1.25rem}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--muted)!important}[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stWidgetLabel"],[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *{color:#D1D5DB!important;font-size:.78rem;font-weight:500}[data-testid="stSidebar"] hr{border-color:var(--line)}
[data-testid="stSidebar"] [data-baseweb="select"],[data-testid="stSidebar"] [data-baseweb="select"] *,[data-testid="stSidebar"] input{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:var(--panel)!important;border-color:var(--line)!important;border-radius:6px!important}[data-testid="stSidebar"] [data-baseweb="tag"]{background:rgba(10,232,182,.12)!important;border:1px solid rgba(10,232,182,.25)!important;color:#C8FFF0!important}[data-testid="stSidebar"] [data-baseweb="tag"] *{color:#C8FFF0!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:var(--panel-raised)!important;color:var(--ink)!important}[data-baseweb="popover"] *,[data-baseweb="menu"] *,[role="listbox"] *,[role="option"] *{color:var(--ink)!important}[data-baseweb="popover"] [aria-selected="true"],[role="option"][aria-selected="true"]{background:rgba(10,232,182,.12)!important}
.stCaption,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}.stTabs [data-baseweb="tab-list"]{gap:1.3rem;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"],[data-baseweb="tab"] *{color:var(--muted)!important;padding:0 0 .72rem;font-family:'Inter',sans-serif;font-size:.84rem;transition:color .16s ease}.stTabs [data-baseweb="tab"]:hover,[data-baseweb="tab"]:hover *{color:var(--ink)!important}.stTabs [data-baseweb="tab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"] *{color:var(--mint)!important;font-weight:600}.stTabs [data-baseweb="tab-highlight"]{background-color:var(--mint)!important;height:2px}
button[kind="primary"],.stDownloadButton button{border-radius:6px!important;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}.stDownloadButton button{background:rgba(10,232,182,.1)!important;border:1px solid rgba(10,232,182,.32)!important;color:#C8FFF0!important}.stDownloadButton button:hover{background:rgba(10,232,182,.17)!important;border-color:var(--mint)!important;box-shadow:0 0 16px rgba(10,232,182,.12);transform:translateY(-1px)}
div[data-testid="stPlotlyChart"]{background:linear-gradient(145deg,rgba(17,24,39,.82),rgba(11,15,23,.72));border:1px solid var(--line);border-radius:10px;padding:.45rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.035);transition:border-color .18s ease,box-shadow .18s ease}div[data-testid="stPlotlyChart"]:hover{border-color:rgba(6,182,212,.32);box-shadow:0 8px 24px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.05)}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:9px;overflow:hidden}[data-testid="stAlert"]{border-radius:8px}
@media (min-width:769px) and (max-width:1100px){.block-container{padding:1.5rem 1.6rem 2.5rem}}
@media (max-width:768px){.block-container{padding:1rem .85rem 2rem}.hero{padding:.35rem 0 1rem}.hero-top{align-items:flex-start;flex-direction:column;gap:.55rem}.hero h1{font-size:clamp(1.65rem,7vw,2.2rem);line-height:1.12}.hero p{font-size:.88rem}.stTabs [data-baseweb="tab-list"]{gap:.8rem;overflow-x:auto}.stTabs [data-baseweb="tab"]{font-size:.78rem;white-space:nowrap;padding:0 0 .65rem}[data-testid="stMetric"]{padding:.8rem;border-radius:8px}div[data-testid="stPlotlyChart"]{padding:.2rem;border-radius:8px}}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;scroll-behavior:auto!important}}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# ◆ PARCL")
st.sidebar.caption("REAL ESTATE INTELLIGENCE")
st.sidebar.divider()
st.sidebar.subheader("Explore buyers")

@st.cache_data(show_spinner="Loading source files…")
def get_features():
    clients, properties = load_data(ROOT)
    return build_buyer_features(clients, properties)

try:
    data = get_features()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load buyer data: {exc}")
    st.info("Place non-empty clients.csv and properties.csv files beside app.py, with client_id and client_ref columns.")
    st.stop()

filter_fields = [("country", "Country"), ("region", "Region"), ("acquisition_purpose", "Acquisition purpose"), ("client_type", "Client type")]
filters = {}
for field, label in filter_fields:
    values = sorted(data[field].dropna().astype(str).unique()) if field in data else []
    filters[field] = st.sidebar.multiselect(label, values, placeholder="All")

filtered = data.copy()
for field, selections in filters.items():
    if selections:
        filtered = filtered[filtered[field].astype(str).isin(selections)]

st.sidebar.divider()
st.sidebar.subheader("Model settings")
method_label = st.sidebar.selectbox("Clustering method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of segments", min_value=2, max_value=8, value=4)
st.sidebar.caption("Features are standardized; identifiers and direct personal identifiers are excluded.")

st.markdown(f'<div class="hero"><div class="hero-top"><div class="eyebrow">PARCL · BUYER INTELLIGENCE</div><div class="model-chip">ACTIVE RUN · {method_label.upper()} · {n_clusters} SEGMENTS</div></div><h1>Real Estate Buyer Intelligence</h1><p>Explore observed buyer patterns, compare segment behavior, and review geographic distribution from the available source data.</p></div>', unsafe_allow_html=True)
if filtered.empty:
    st.warning("No buyers match these filters. Adjust or clear one or more selections in the sidebar.")
    st.stop()
if len(filtered) <= n_clusters:
    st.warning(f"The current filters return {len(filtered)} buyer(s). Select fewer segments or broaden the filters; clustering needs more buyers than segments.")
    st.stop()

model_input = filtered.drop(columns=["client_id"], errors="ignore")
try:
    method = "kmeans" if method_label == "K-Means" else "hierarchical"
    labels, model, matrix = cluster_buyers(model_input, make_preprocessor(model_input), method, n_clusters)
    results = filtered.copy()
    results["segment"] = labels.astype(str)
    scores = evaluate_clusters(matrix, labels)
    profiles = summarize_segments(results)
except (ValueError, TypeError, FloatingPointError) as exc:
    st.error(f"Could not produce segments for the current data and settings: {exc}")
    st.stop()

navy, blue, gold = "#0AE8B6", "#06B6D4", "#F59E0B"
segment_palette = [navy, blue, "#818CF8", gold, "#34D399", "#A78BFA", "#22D3EE", "#FB7185"]
plot_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9CA3AF"),
    legend=dict(font=dict(color="#D1D5DB"), title=dict(font=dict(color="#F9FAFB"))),
    xaxis=dict(color="#9CA3AF", tickfont=dict(color="#9CA3AF"), title_font=dict(color="#D1D5DB"), gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
    yaxis=dict(color="#9CA3AF", tickfont=dict(color="#9CA3AF"), title_font=dict(color="#D1D5DB"), gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#06B6D4", font=dict(color="#F9FAFB", family="Inter")),
    margin=dict(l=18,r=18,t=25,b=20), legend_title_text="",
)
tabs = st.tabs(["Segmentation Overview", "Investor Behavior", "Geographic Analysis", "Segment Insights"])

with tabs[0]:
    st.caption("A current view of the buyers returned by your filters.")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Buyers in view", f"{len(results):,}")
    m2.metric("Segments", f"{results.segment.nunique()}")
    m3.metric("Silhouette score", f"{scores['silhouette']:.3f}" if pd.notna(scores["silhouette"]) else "Unavailable", help="Higher indicates better separation.")
    m4.metric("Davies–Bouldin", f"{scores['davies_bouldin']:.3f}" if pd.notna(scores["davies_bouldin"]) else "Unavailable", help="Lower indicates more compact, separated clusters.")
    counts=results.segment.value_counts().rename_axis("Segment").reset_index(name="Buyers")
    counts["Segment"]=counts["Segment"].map(lambda s:f"Segment {int(s)+1}")
    a,b=st.columns([1,1.3])
    with a:
        fig=px.pie(counts,names="Segment",values="Buyers",hole=.68,color_discrete_sequence=segment_palette)
        fig.update_layout(**plot_layout,showlegend=True); fig.update_traces(textposition="outside",textinfo="percent",textfont_color="#F9FAFB")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.bar(counts,x="Segment",y="Buyers",color="Segment",color_discrete_sequence=segment_palette,text="Buyers")
        fig.update_layout(**plot_layout,showlegend=False,xaxis_title="",yaxis_title="Buyers"); fig.update_traces(textposition="outside",marker_line_width=0,textfont_color="#F9FAFB")
        st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.caption("Observed acquisition preferences and transaction behavior. Missing values remain missing in the source.")
    c1,c2=st.columns(2)
    if "acquisition_purpose" in results:
        purpose=results.groupby(["segment","acquisition_purpose"],dropna=False).size().reset_index(name="Buyers")
        purpose["segment"]=purpose["segment"].map(lambda s:f"Segment {int(s)+1}")
        with c1:
            fig=px.bar(purpose,x="segment",y="Buyers",color="acquisition_purpose",barmode="stack",color_discrete_sequence=[navy,blue,gold,"#818CF8"])
            fig.update_layout(**plot_layout,xaxis_title="",yaxis_title="Buyers",legend_title="Purpose"); st.plotly_chart(fig,use_container_width=True)
    if "mean_sale_price" in results:
        price=results.groupby("segment",as_index=False)["mean_sale_price"].mean().dropna()
        price["segment"]=price["segment"].map(lambda s:f"Segment {int(s)+1}")
        with c2:
            if not price.empty:
                fig=px.bar(price,x="segment",y="mean_sale_price",color="segment",color_discrete_sequence=segment_palette)
                fig.update_layout(**plot_layout,showlegend=False,xaxis_title="",yaxis_title="Mean recorded sale price ($)"); st.plotly_chart(fig,use_container_width=True)
            else: st.info("No recorded sale prices are available for these buyers.")
    if "loan_applied" in results:
        loan=results.groupby(["segment","loan_applied"],dropna=False).size().reset_index(name="Buyers")
        loan["segment"]=loan["segment"].map(lambda s:f"Segment {int(s)+1}")
        fig=px.bar(loan,x="segment",y="Buyers",color="loan_applied",barmode="group",color_discrete_sequence=[blue,navy])
        fig.update_layout(**plot_layout,xaxis_title="",yaxis_title="Buyers",legend_title="Loan applied"); st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    st.caption("Buyer counts by recorded client geography. The source does not provide property coordinates.")
    geo_col="region" if "region" in results else "country"
    geo=results.groupby([geo_col,"segment"],dropna=False).size().reset_index(name="Buyers").sort_values("Buyers",ascending=False)
    geo["segment"]=geo["segment"].map(lambda s:f"Segment {int(s)+1}")
    if geo.empty: st.info("No geographic fields are available in the filtered data.")
    else:
        fig=px.bar(geo.head(50),x="Buyers",y=geo_col,color="segment",orientation="h",barmode="stack",color_discrete_sequence=segment_palette)
        geo_layout={**plot_layout,"yaxis":{**plot_layout["yaxis"],"categoryorder":"total ascending"}}
        fig.update_layout(**geo_layout,height=max(380,min(720,24*len(geo.head(25)))),xaxis_title="Buyers",yaxis_title=""); st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    st.caption("Segment summaries use only measured fields. A segment label describes a model grouping, not a factual buyer identity.")
    display=profiles.rename(columns={"segment":"Segment","buyers":"Buyers","share_of_buyers_pct":"Share of buyers (%)"})
    display["Segment"]=display["Segment"].map(lambda s:f"Segment {int(s)+1}")
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.download_button("Download segmented buyer summary (CSV)",results.to_csv(index=False).encode("utf-8"),"parcl_buyer_segments.csv","text/csv")
    st.caption("Clustering is exploratory and depends on the current filters, chosen method, and number of segments.")
