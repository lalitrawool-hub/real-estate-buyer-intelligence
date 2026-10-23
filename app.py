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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--navy:#13283f;--blue:#8eb4cf;--blue-dark:#527f9f;--warm:#faf9f6;--gold:#d8ba70;--beige:#eee9df;--ink:#213349;--muted:#788696}
.stApp{background:var(--warm);color:var(--ink);font-family:'DM Sans',sans-serif}.block-container{padding:2.2rem 3.5rem 3rem;max-width:1580px}
h1,h2,h3{font-family:'Manrope',sans-serif;color:var(--navy);letter-spacing:-.035em}.hero{padding:1rem 0 1.6rem}.eyebrow{color:#a0874e;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}.hero p{color:var(--muted);font-size:1.02rem;margin:.35rem 0 0}
[data-testid="stMetric"]{background:white;border:1px solid #ece8e0;border-radius:18px;padding:1rem 1.2rem;box-shadow:0 5px 18px rgba(19,40,63,.045)}[data-testid="stMetricLabel"]{color:#788696}[data-testid="stMetricValue"]{color:#13283f;font-family:Manrope}
[data-testid="stSidebar"]{background:#f1eee7;border-right:1px solid #e8e2d7}[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2{color:var(--navy)}
.section-note{color:#788696;font-size:.91rem}.stTabs [data-baseweb="tab-list"]{gap:1.2rem;border-bottom:1px solid #e8e4dc}.stTabs [data-baseweb="tab"]{color:#657587;padding:0 0 .8rem}.stTabs [aria-selected="true"]{color:#13283f}
div[data-testid="stPlotlyChart"]{background:white;border:1px solid #ece8e0;border-radius:18px;padding:.5rem;box-shadow:0 5px 18px rgba(19,40,63,.035)}
@media (min-width:769px) and (max-width:1100px){.block-container{padding:2rem 2rem 2.5rem}}
@media (max-width:768px){.block-container{padding:1.25rem 1rem 2rem}.hero{padding:.4rem 0 1rem}.hero h1{font-size:clamp(1.75rem,7vw,2.45rem);line-height:1.1}.hero p{font-size:.94rem}.stTabs [data-baseweb="tab-list"]{gap:.7rem;overflow-x:auto}.stTabs [data-baseweb="tab"]{font-size:.82rem;white-space:nowrap;padding:0 0 .65rem}[data-testid="stMetric"]{padding:.8rem;border-radius:14px}div[data-testid="stPlotlyChart"]{padding:.25rem;border-radius:14px}}
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

st.markdown('<div class="hero"><div class="eyebrow">PARCL · MARKET INTELLIGENCE</div><h1>Real Estate Buyer Intelligence</h1><p>Understand buyer behavior, discover meaningful segments, and explore where demand is taking shape.</p></div>', unsafe_allow_html=True)
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

navy, blue, gold = "#18334f", "#8eb4cf", "#d8ba70"
plot_layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#536477"), margin=dict(l=18,r=18,t=25,b=20), legend_title_text="")
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
        fig=px.pie(counts,names="Segment",values="Buyers",hole=.68,color_discrete_sequence=[navy,blue,gold,"#c8b9a4","#7191a8"])
        fig.update_layout(**plot_layout,showlegend=True); fig.update_traces(textposition="outside",textinfo="percent")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.bar(counts,x="Segment",y="Buyers",color="Segment",color_discrete_sequence=[navy,blue,gold,"#c8b9a4"],text="Buyers")
        fig.update_layout(**plot_layout,showlegend=False,xaxis_title="",yaxis_title="Buyers"); fig.update_traces(textposition="outside",marker_line_width=0)
        st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.caption("Observed acquisition preferences and transaction behavior. Missing values remain missing in the source.")
    c1,c2=st.columns(2)
    if "acquisition_purpose" in results:
        purpose=results.groupby(["segment","acquisition_purpose"],dropna=False).size().reset_index(name="Buyers")
        purpose["segment"]=purpose["segment"].map(lambda s:f"Segment {int(s)+1}")
        with c1:
            fig=px.bar(purpose,x="segment",y="Buyers",color="acquisition_purpose",barmode="stack",color_discrete_sequence=[navy,blue,gold])
            fig.update_layout(**plot_layout,xaxis_title="",yaxis_title="Buyers",legend_title="Purpose"); st.plotly_chart(fig,use_container_width=True)
    if "mean_sale_price" in results:
        price=results.groupby("segment",as_index=False)["mean_sale_price"].mean().dropna()
        price["segment"]=price["segment"].map(lambda s:f"Segment {int(s)+1}")
        with c2:
            if not price.empty:
                fig=px.bar(price,x="segment",y="mean_sale_price",color="segment",color_discrete_sequence=[navy,blue,gold,"#c8b9a4"])
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
        fig=px.bar(geo.head(50),x="Buyers",y=geo_col,color="segment",orientation="h",barmode="stack",color_discrete_sequence=[navy,blue,gold,"#c8b9a4"])
        fig.update_layout(**plot_layout,height=max(380,min(720,24*len(geo.head(25)))),yaxis={"categoryorder":"total ascending"},xaxis_title="Buyers",yaxis_title=""); st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    st.caption("Segment summaries use only measured fields. A segment label describes a model grouping, not a factual buyer identity.")
    display=profiles.rename(columns={"segment":"Segment","buyers":"Buyers","share_of_buyers_pct":"Share of buyers (%)"})
    display["Segment"]=display["Segment"].map(lambda s:f"Segment {int(s)+1}")
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.download_button("Download segmented buyer summary (CSV)",results.to_csv(index=False).encode("utf-8"),"parcl_buyer_segments.csv","text/csv")
    st.caption("Clustering is exploratory and depends on the current filters, chosen method, and number of segments.")
