from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scanpy as sc
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
IMAGE_DIR = APP_DIR / "images"

st.set_page_config(
    page_title="BioPoplar · Populus Single-Cell & Spatial Atlas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASETS = {
    "Spatial RNA-seq": {
        "Axillary bud": {
            "file": "spatial_bud.h5ad",
            "description": "Serial sections spanning axillary-bud developmental domains.",
        },
        "Shoot apical meristem": {
            "file": "spatial_sam.h5ad",
            "description": "Spatial organization of meristematic and developing leaf tissues.",
        },
        "Petiole · cross section": {
            "file": "spatial_petiole_cross.h5ad",
            "description": "Cross-sectional atlas of petiole tissue and vascular domains.",
        },
        "Petiole · longitudinal": {
            "file": "spatial_petiole_longitudinal.h5ad",
            "description": "Longitudinal view of petiole development and tissue patterning.",
        },
        "Stem · cross section": {
            "file": "spatial_stem_cross.h5ad",
            "description": "Cross-sectional atlas of primary and secondary stem tissues.",
        },
    },
    "Single-cell RNA-seq": {
        "Leaf · stage 1": {"file": "sobj_Leaf_1_slim.h5ad", "description": "Leaf stage 1."},
        "Leaf · stage 3": {"file": "sobj_Leaf_3_slim.h5ad", "description": "Leaf stage 3."},
        "Leaf · stage 5": {"file": "sobj_Leaf_5_slim.h5ad", "description": "Leaf stage 5."},
        "Primary stem": {"file": "sobj_Primary_stem_slim.h5ad", "description": "Primary stem."},
        "Secondary stem": {"file": "sobj_Secondary_stem_slim.h5ad", "description": "Secondary stem."},
        "Shoot apical meristem": {"file": "sobj_SAM_slim.h5ad", "description": "Shoot apical meristem."},
    },
}

PALETTE = [
    "#2F6B4F", "#D39B36", "#56799A", "#9A5E62", "#6F8E4E", "#7A6A9D",
    "#C66A3D", "#3C8D8B", "#A37A46", "#51646F", "#B65F7A", "#7899C0",
]
EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "biopoplar_expression", ["#F3F6F0", "#B9D7B0", "#4A8C5B", "#173E2C"]
)


def inject_css():
    st.markdown(
        """
        <style>
        :root { --forest:#173E2C; --leaf:#2F6B4F; --sage:#E8F0E8; --ink:#24312B; --gold:#C9973C; }
        .stApp { background: #F7F8F5; color: var(--ink); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #EEF3EC 0%, #F8F9F6 100%); border-right: 1px solid #DCE5D9; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #34473D; }
        .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--forest); letter-spacing: -0.02em; }
        h1 { font-weight: 720; }
        .atlas-kicker { color: var(--leaf); font-size: .78rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
        .atlas-hero { padding: 1.2rem 0 1rem; border-bottom: 1px solid #DCE5D9; margin-bottom: 1.4rem; }
        .atlas-hero h1 { font-size: clamp(2.1rem, 4vw, 3.65rem); margin: .15rem 0 .35rem; line-height: 1.03; }
        .atlas-hero p { max-width: 780px; color: #58665F; font-size: 1.05rem; line-height: 1.65; }
        .dataset-note { background:#FFFFFF; border:1px solid #DEE7DC; border-left:4px solid var(--gold); border-radius:10px; padding:.85rem 1rem; color:#4C5B53; }
        [data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E0E7DE; border-radius:12px; padding:.8rem 1rem; box-shadow:0 3px 14px rgba(31,61,43,.04); }
        [data-testid="stMetricLabel"] { color:#607068; }
        [data-testid="stMetricValue"] { color:var(--forest); }
        div[data-baseweb="select"] > div { background:#FFFFFF; border-color:#CBD8C8; }
        .stButton > button { background:var(--forest); color:white; border:none; border-radius:8px; }
        .stTabs [data-baseweb="tab-list"] { gap:1.25rem; border-bottom:1px solid #DCE5D9; }
        .stTabs [data-baseweb="tab"] { color:#627168; padding:.7rem .2rem; }
        .stTabs [aria-selected="true"] { color:var(--forest); font-weight:700; border-bottom-color:var(--forest); }
        .mini-label { color:#6C7A72; font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False, max_entries=2)
def load_dataset(path_string):
    return sc.read_h5ad(path_string)


def available_libraries():
    # Preserve the original scRNA-seq catalogue even when its large files are
    # mounted only in production. The spatial atlases are strictly additive.
    order = ["Single-cell RNA-seq", "Spatial RNA-seq"]
    return {library: DATASETS[library] for library in order}


def obs_column(adata, candidates):
    return next((name for name in candidates if name in adata.obs.columns), None)


def expression_vector(adata, gene):
    values = adata[:, gene].X
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def category_colors(values):
    categories = pd.Categorical(values)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(categories.categories))]
    return categories, dict(zip(categories.categories, colors))


def spatial_figure(adata, section_col, section, color_by, celltype_col, gene):
    mask = np.asarray(adata.obs[section_col].astype(str) == str(section))
    coords = np.asarray(adata.obsm["spatial"])[mask]
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="#FFFFFF")

    # Match Scanpy's Visium convention: spot coordinates are full-resolution
    # pixels and the embedded tissue image is displayed using its lowres scale.
    spatial_entry = adata.uns.get("spatial", {}).get(str(section), {})
    tissue_image = spatial_entry.get("images", {}).get("lowres")
    lowres_scale = float(
        spatial_entry.get("scalefactors", {}).get("tissue_lowres_scalef", 1.0)
    )
    has_tissue_image = tissue_image is not None
    plot_coords = coords * lowres_scale if has_tissue_image else coords
    if has_tissue_image:
        ax.imshow(tissue_image, origin="upper")

    if color_by == "Gene expression":
        values = expression_vector(adata, gene)[mask]
        order = np.argsort(values)
        scatter = ax.scatter(
            plot_coords[order, 0], plot_coords[order, 1], c=values[order], cmap=EXPRESSION_CMAP,
            s=20, linewidths=0, alpha=.9, rasterized=True,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=.72, pad=.025)
        cbar.set_label(f"Processed expression · {gene}", color="#415047")
        title = gene
    else:
        categories, color_map = category_colors(adata.obs.loc[mask, celltype_col].astype(str))
        for category in categories.categories:
            selected = np.asarray(categories == category)
            ax.scatter(
                plot_coords[selected, 0], plot_coords[selected, 1], s=18, linewidths=0,
                color=color_map[category], label=str(category), alpha=.88, rasterized=True,
            )
        ax.legend(
            title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
            fontsize=8, title_fontsize=9, markerscale=1.35,
        )
        title = color_by

    ax.set_title(f"{section} · {title}", loc="left", fontsize=14, color="#173E2C", pad=12)
    ax.set_aspect("equal")
    if not has_tissue_image:
        ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_grouped_expression(adata, genes, groupby, kind):
    if kind == "Violin":
        sc.pl.violin(
            adata, keys=genes, groupby=groupby, rotation=90,
            multi_panel=True, show=False, stripplot=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(max(9, len(genes) * 3.2), 5.5)
    elif kind == "Dot plot":
        result = sc.pl.dotplot(
            adata, genes, groupby=groupby, show=False, cmap=EXPRESSION_CMAP,
            colorbar_title="Mean expression", size_title="Fraction expressed",
        )
        axes = result.get_axes() if hasattr(result, "get_axes") else result
        fig = axes["mainplot_ax"].figure
        fig.set_size_inches(max(8, len(genes) * 1.1), max(5, adata.obs[groupby].nunique() * .32))
    else:
        sc.pl.heatmap(
            adata, genes, groupby=groupby, show=False, swap_axes=True,
            cmap=EXPRESSION_CMAP, dendrogram=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(max(9, adata.obs[groupby].nunique() * .45), max(4, len(genes) * .55))
    fig.patch.set_facecolor("white")
    return fig


def landing_page(libraries):
    st.markdown(
        """
        <div class="atlas-hero">
          <div class="atlas-kicker">Populus developmental genomics</div>
                    <h1>BioPoplar · Populus Single-Cell & Spatial Atlas</h1>
          <p>Explore gene expression and annotated cell identities across poplar tissues using
          spatial RNA-seq and single-cell RNA sequencing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Available datasets", sum(len(x) for x in libraries.values()))
    c2.metric("Spatial tissues", len(libraries.get("Spatial RNA-seq", {})))
    st.markdown("### Start exploring")
    st.markdown(
        '<div class="dataset-note">Choose a data modality and tissue from the sidebar. '
        'Spatial datasets support section-level tissue maps, curated cell-type views, and gene-expression summaries.</div>',
        unsafe_allow_html=True,
    )
    # Landing page illustration(s) — prefer renamed site image, fallback to older name
    if (IMAGE_DIR / "scRNA_workflow.png").exists():
        st.image(str(IMAGE_DIR / "scRNA_workflow.png"), width="stretch")
    if (IMAGE_DIR / "website_pic_spatial.png").exists():
        st.image(str(IMAGE_DIR / "website_pic_spatial.png"), width="stretch")


def main():
    inject_css()
    libraries = available_libraries()

    with st.sidebar:
        if (IMAGE_DIR / "BioPoplar_Logo2.png").exists():
            st.image(str(IMAGE_DIR / "BioPoplar_Logo2.png"), width="stretch")
        st.markdown("### Atlas explorer")
        if not libraries:
            st.error("No H5AD datasets were found in the data directory.")
            return
        library = st.selectbox("Data modality", list(libraries), index=0)
        tissue_options = ["Select a tissue…"] + list(libraries[library])
        tissue = st.selectbox("Tissue atlas", tissue_options, key="tissue")
        st.caption("Processed expression matrices · Populus spatial and single-cell atlas")

    if tissue == "Select a tissue…":
        landing_page(libraries)
        return

    spec = libraries[library][tissue]
    dataset_path = DATA_DIR / spec["file"]
    if not dataset_path.exists():
        st.error(
            f"The configured dataset `{spec['file']}` is not available in this local data directory. "
            "Its original scRNA-seq catalogue entry has been preserved for the production data mount."
        )
        return
    with st.spinner(f"Loading {tissue}…"):
        adata = load_dataset(str(dataset_path))

    celltype_col = obs_column(adata, ["cell_type_name", "celltypes_v2", "celltypes", "celltype"])
    section_col = obs_column(adata, ["section", "slice_id", "sample"])

    st.markdown(
        f'<div class="atlas-kicker">{library}</div><h1>{tissue}</h1>'
        f'<div class="dataset-note">{spec["description"]}</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spots / cells", f"{adata.n_obs:,}")
    m2.metric("Genes", f"{adata.n_vars:,}")
    m3.metric("Cell types", f"{adata.obs[celltype_col].nunique():,}" if celltype_col else "—")
    m4.metric("Sections", f"{adata.obs[section_col].nunique():,}" if section_col else "—")

    spatial_ready = library == "Spatial RNA-seq" and "spatial" in adata.obsm and section_col
    tabs = st.tabs(["Spatial explorer", "Expression summary", "Dataset information"] if spatial_ready
                   else ["Embedding explorer", "Expression summary", "Dataset information"])

    with tabs[0]:
        if spatial_ready:
            controls, display = st.columns([1, 2.35], gap="large")
            with controls:
                st.markdown("#### Map controls")
                sections = sorted(adata.obs[section_col].astype(str).unique())
                section = st.selectbox("Section", sections)
                choices = ["Cell type"] if celltype_col else []
                choices.append("Gene expression")
                color_by = st.radio("Color spots by", choices)
                gene = None
                if color_by == "Gene expression":
                    gene = st.selectbox("Gene", adata.var_names.tolist(), index=0)
                n_spots = int((adata.obs[section_col].astype(str) == section).sum())
                st.caption(f"{n_spots:,} spots in this section")
            with display:
                fig = spatial_figure(adata, section_col, section, color_by, celltype_col, gene)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        elif "X_umap" in adata.obsm:
            if celltype_col:
                fig = sc.pl.umap(adata, color=celltype_col, show=False, return_fig=True)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.info("This dataset has no cell-type annotation available for coloring.")
        else:
            st.info("This compact dataset does not contain spatial coordinates or an embedding.")

    with tabs[1]:
        left, right = st.columns([1, 2.35], gap="large")
        with left:
            st.markdown("#### Expression controls")
            genes = st.multiselect("Genes", adata.var_names.tolist(), max_selections=20)
            plot_kind = st.radio("Plot type", ["Violin", "Dot plot", "Heatmap"], horizontal=True)
            st.caption("All summaries are grouped by the curated cell-type annotation.")
        with right:
            if genes and celltype_col:
                fig = plot_grouped_expression(adata, genes, celltype_col, plot_kind)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.info("Select one or more genes to generate an expression summary.")

    with tabs[2]:
        st.markdown("#### Dataset contents")
        st.write(
            "Expression values are processed, thresholded at 0.5, stored as a sparse float32 matrix, "
            "and filtered to genes detected in at least three observations. Raw counts are not included."
        )
        summary = pd.DataFrame({
            "Field": ["File", "Matrix", "Annotations", "Spatial coordinates", "Raw counts"],
            "Value": [
                spec["file"], f"{adata.n_obs:,} × {adata.n_vars:,} sparse expression",
                celltype_col or "None",
                "Available" if "spatial" in adata.obsm else "Not included", "Not included",
            ],
        })
        st.dataframe(summary, hide_index=True, width="stretch")
        st.markdown("#### Citation")
        st.info("Citation and manuscript DOI will be added upon publication.")

    st.markdown("---")
    st.caption("BioPoplar Atlas · University of Georgia · For research use")


if __name__ == "__main__":
    main()
