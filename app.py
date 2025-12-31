import streamlit as st
import os
import pandas as pd
import networkx as nx
import cpnet
import altair as alt
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict

# CONSTANT
DATA_DIR = "data"

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Core–Periphery Structure in Developer Collaboration Networks",
    layout="wide"
)

# FUNCTION
def clear_all_session_state():
    st.session_state.clear()

def process_uploaded_files(uploaded_files):
    """
    Processes the files uploaded by the user, validating names and column structures.
    """
    uploaded_data = {}
    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        # Validation: check filename pattern edges_list_{version}.csv
        if fname.startswith("edges_list_") and fname.endswith(".csv"):
            try:
                version_name = fname.replace("edges_list_", "").replace(".csv", "")
                df = pd.read_csv(uploaded_file)
                
                # Column validation: Source, Target, and Weight must exist
                required_cols = {'Source', 'Target', 'Weight'}
                if required_cols.issubset(df.columns):
                    uploaded_data[version_name] = df
                else:
                    missing = required_cols - set(df.columns)
                    st.error(f"File {fname} is missing columns: {missing}")
                    st.stop()
            except Exception as e:
                st.error(f"Error reading {fname}: {e}")
                st.stop()
        else:
            st.warning(f"Filename {fname} invalid. Use 'edges_list_VERSION.csv' format.")
            st.stop()
            
    return uploaded_data

@st.cache_data
def load_all_network_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all CSV files from the data/ directory containing edge lists.
    Returns a dictionary where the key is the version name (e.g., '1_0').
    """
    all_data = {}
    if not os.path.exists(data_dir):
        # Create the directory if it doesn't exist to prevent errors later
        os.makedirs(data_dir, exist_ok=True)
        st.error(f"Data directory not found or empty at: {data_dir}. Place CSV files here.")
        return all_data

    for filename in os.listdir(data_dir):
        if filename.startswith("edges_list_") and filename.endswith(".csv"):
            try:
                # Extract version from filename (e.g., 'edges_list_1_0.csv' -> '1_0')
                version_name = filename.replace("edges_list_", "").replace(".csv", "")
                
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                
                # Ensure standard columns are present
                if 'Source' in df.columns and 'Target' in df.columns and 'Weight' in df.columns:
                    all_data[version_name] = df
                else:
                    st.warning(f"File {filename} skipped: Missing Source, Target, or Weight columns.")
            except Exception as e:
                st.error(f"Failed to load or process {filename}: {e}")
                
    return all_data

@st.cache_data(show_spinner="Building Graphs and Calculating Rombach core_score...")
def process_all_graphs(all_data: Dict[str, pd.DataFrame]) -> Dict[str, nx.Graph]:
    """
    Builds NetworkX Graphs, calculates Rombach core_score, and stores the core_score 
    as a node attribute for ALL snapshots. Caching is based entirely on the input DataFrames.
    """
    all_graphs = {}
    
    for version, df_edges in all_data.items():
        # 1. Build the full graph using nx.Graph for undirected co-authorship network
        G = nx.from_pandas_edgelist(
            df_edges, 
            source='Source', 
            target='Target', 
            edge_attr='Weight', 
            create_using=nx.Graph()
        )

        # # 2. LCC Filter Logic
        # if lcc_only:
        #     # Get all connected components sorted by size
        #     components = list(nx.connected_components(G_full))
        #     num_components = len(components)
            
        #     if num_components > 1:
        #         # Select the largest component
        #         lcc_nodes = max(components, key=len)
        #         G = G_full.subgraph(lcc_nodes).copy()
                
        #         # Calculate how many nodes were removed
        #         removed_nodes = G_full.number_of_nodes() - G.number_of_nodes()
        #         st.sidebar.info(f"[{version.replace('_', '.')}] LCC Active: Found {num_components} components. Removed {removed_nodes} nodes from {num_components - 1} smaller clusters.")
        #     else:
        #         G = G_full
        # else:
        #     G = G_full
        
        if G.number_of_nodes() < 2:
            all_graphs[version] = G
            continue
        
        # 3. Calculate Rombach core_score 
        try:
            alg = cpnet.Rombach(alpha=0, beta=0.8, num_runs=10)
            alg.detect(G)
            core_score_dict = alg.get_coreness()
            pair_id = alg.get_pair_id()
            # st.write(pair_id)

            # sig_c, sig_x, significant, p_values = cpnet.qstest(
            #     pair_id, core_score_dict, G, alg, significance_level=0.05, num_of_thread=1
            # )
            # st.write(sig_c)
            # st.write(sig_x)
            # st.write(significant)
            # st.write(p_values)
            
            # 3. Set core_score as Node Attribute
            nx.set_node_attributes(G, core_score_dict, 'rombach_core_score') 
            nx.set_node_attributes(G, pair_id, 'pair_id') 
            
        except Exception as e:
            st.warning(f"Failed to calculate core_score for {version}: {e}")

            
        all_graphs[version] = G
        
    st.success(f"Successfully processed {len(all_graphs)} NetworkX graphs.")
    return all_graphs, alg


def render_network_size_section(df_overview: pd.DataFrame):
    """Renders the double line chart of network size."""
    
    df_melted = df_overview.melt(
        id_vars=['Version'], 
        value_vars=['Num_Developers', 'Num_Edges'],
        var_name='Metric',
        value_name='Count'
    )
    label_map = {
        'Num_Developers': 'Number of Developers', 
        'Num_Edges': 'Number of Edges'
    }
    df_melted['Legend_Label'] = df_melted['Metric'].map(label_map)
    # Base chart
    base = alt.Chart(df_melted).encode(
        # X-axis: Versi, diurutkan sesuai urutan dalam DataFrame
        x=alt.X(
            'Version', 
            sort=df_overview['Version'].tolist(), 
            title='Release Version',
            axis=alt.Axis(labelAngle=0) 
        ),
        # Y-axis: Count
        y=alt.Y('Count', title='Number of Nodes / Edges'),
        # Color based on Metric
        color=alt.Color('Legend_Label', title='Metric')
    ).properties(
        title='Evolution of Network Size across Release Versions'
    )
    # Line Chart
    lines = base.mark_line().encode(
        # Tambahkan tooltip untuk menampilkan detail saat hover
        tooltip=['Version', 'Metric', 'Count']
    )
    # Point Chart (Marker)
    points = base.mark_point(
        filled=True, 
        size=50
    ).encode(
        tooltip=['Version', 'Metric', 'Count']
    )
    
    # Merge Line dan Point Chart
    chart = (lines + points).interactive() 
    
    st.altair_chart(chart, use_container_width=True)

def render_network_visualization(all_graphs: Dict[str, nx.Graph], sorted_versions: list):
    """
    Renders two side-by-side interactive network graphs:
    1. Original network with core_score-based styling.
    2. Simulated network after removing the top developer.
    """
    st.subheader("Core–Periphery Network Visualization")

    # Version selection
    version_options = [v.replace('_', '.') for v in sorted_versions]
    selected_ver_str = st.selectbox("Select Version to Visualize:", version_options, index=len(version_options)-1, key="viz_ver_select")
    
    # Map back to original key
    selected_key = selected_ver_str.replace('.', '_')
    G_orig = all_graphs[selected_key].copy()
    
    if G_orig.number_of_nodes() == 0:
        st.warning("The selected graph has no nodes.")
        return

    # Identify top developer for removal
    core_score_attr = nx.get_node_attributes(G_orig, 'rombach_core_score')
    top_dev = max(core_score_attr, key=core_score_attr.get) if core_score_attr else None
    
    G_sim = G_orig.copy()
    if top_dev:
        G_sim.remove_node(top_dev)

    col1, col2 = st.columns(2)

    def build_agraph_elements(G):
        nodes = []
        edges = []
        
        # --- 1. PRE-CALCULATION
        # Get all core score and weight
        all_core_score = [d.get('rombach_core_score', 0) for _, d in G.nodes(data=True)]
        all_weights = [d.get('Weight', 1) for _, _, d in G.edges(data=True)]

        # Min/Max core_score
        if all_core_score:
            min_c, max_c = min(all_core_score), max(all_core_score)
        else:
            min_c, max_c = 0, 1

        # Min/Max Weight
        if all_weights:
            min_w, max_w = min(all_weights), max(all_weights)
        else:
            min_w, max_w = 1, 1

        # --- 2. NODE LOGIC ---
        for node, attrs in G.nodes(data=True):
            core_score = attrs.get('rombach_core_score', 0)
            
            min_node_size = 10
            max_node_size = 40
            
            if max_c > min_c:
                node_size = min_node_size + (core_score - min_c) * (max_node_size - min_node_size) / (max_c - min_c)
            else:
                node_size = min_node_size
                
            color = f"rgba(31, 119, 180, {0.3 + (core_score * 0.7)})"

            hover_info = f"Developer: {node}\nCore Score: {core_score:.4f}"

            nodes.append(Node(id=node, label=node, size=node_size, color=color, title=hover_info))
            
        # --- 3. EDGE LOGIC ---
        for source, target, attrs in G.edges(data=True):
            weight = attrs.get('Weight', 1)
            
            min_edge_w = 3
            max_edge_w = 12
            
            if max_w > min_w:
                edge_width = min_edge_w + (weight - min_w) * (max_edge_w - min_edge_w) / (max_w - min_w)
            else:
                edge_width = min_edge_w
                
            edges.append(Edge(
                source=source, 
                target=target, 
                color="#D3D3D3", 
                width=edge_width
            ))
            
        return nodes, edges

    # Configuration for ForceAtlas2
    config = Config(
        width=600,
        height=500,
        directed=False,
        physics=True,
        hierarchical=False,
        stabilization={
            "enabled": True,
            "iterations": 1000,
            "fit": True,
        },
        saveLayout=True,
        graphviz_layout="neato", # Alternative base
        forceAtlas2Based={
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
            "avoidOverlap": 1
        }
    )

    with col1:
        st.markdown(f"**Original Collaboration Network (Version {selected_ver_str})**")
        nodes_orig, edges_orig = build_agraph_elements(G_orig)
        agraph(nodes=nodes_orig, edges=edges_orig, config=config)
        st.caption("Node size and opacity represent continuous Rombach core scores.")

    with col2:
        st.markdown(f"**Network after Core Node Removal (Without {top_dev})**")
        nodes_sim, edges_sim = build_agraph_elements(G_sim)
        agraph(nodes=nodes_sim, edges=edges_sim, config=config)
        st.caption(f"Structure after simulated removal of the highest core-score developer.")

def render_top_developers_comparison(df_core_score_full: pd.DataFrame, sorted_versions: list):
    """
    Renders two side-by-side bar charts to compare top developers.
    Defaults to the latest version for the second chart.
    """

    available_versions = [v.replace('_', '.') for v in sorted_versions]
    
    # Set default to latest and previous versions
    latest_idx = len(available_versions) - 1
    prev_idx = max(0, latest_idx - 1)

    col1, col2 = st.columns(2)

    # Reusable function for consistent chart styling
    def create_bar(version_str):
        df_ver = df_core_score_full[df_core_score_full['Version'] == version_str]
        df_top = df_ver.nlargest(10, 'Rombach_Core_Score')
        
        return alt.Chart(df_top).mark_bar(color='#1f77b4').encode(
            y=alt.Y('Developer:N', sort='-x', title=None),
            x=alt.X('Rombach_Core_Score:Q', 
                    title='Rombach Core Score',
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=['Developer', 'Rombach_Core_Score']
        ).properties(
            title=f"Top 10 Developers by Core Score - Version {version_str}",
            height=350
        )

    with col1:
        st.markdown("**Baseline Version**")
        ver1 = st.selectbox("Select Version A:", available_versions, index=prev_idx, key="comp_ver_a")
        st.altair_chart(create_bar(ver1), use_container_width=True)

    with col2:
        st.markdown("**Comparison Version**")
        ver2 = st.selectbox("Select Version B:", available_versions, index=latest_idx, key="comp_ver_b")
        st.altair_chart(create_bar(ver2), use_container_width=True)
   

def render_developer_profile_table(df_core_score_full: pd.DataFrame, sorted_versions: list):
    """
    Transforms core_score data into a wide-format profile table with 
    conditional formatting to highlight trends.
    - Light Blue: Inflow (New entry)
    - Green: Increase
    - Red: Decrease
    - Gray: Stable
    """

    st.subheader("Developer Core Score Profile across Release Versions")
    st.markdown("""
    **Color Legend:**
    - :blue[**Light Blue**]: **New/Re-entry** (First appearance or return after absence)
    - :green[**Green**]: **Increase** (Core score is higher than previous version)
    - :red[**Red**]: **Decrease** (Core score is lower than previous version)
    - :grey[**Gray**]: **Stable** (No change in core score)
    - "None": **Inactive** (No participation in that specific version)
    """)

    # 1. Pivot from long to wide format
    version_cols = [v.replace('_', '.') for v in sorted_versions]
    
    df_wide = df_core_score_full.pivot(
        index='Developer',
        columns='Version',
        values='Rombach_Core_Score'
    )

    # 2. Rename columns to include 'v' prefix (e.g., 'v1.0')
    version_mapping = {v: f"v{v}" for v in version_cols}
    df_wide = df_wide.reindex(columns=version_cols).rename(columns=version_mapping)
    new_version_cols = list(version_mapping.values())

    # 3. Add Presence and sort by it (Descending)
    df_wide['Presence'] = df_wide.notna().sum(axis=1)
    df_wide = df_wide.sort_values(by='Presence', ascending=False)

    # 4. Define styling logic
    def style_trends(row):
        styles = ['' for _ in row]
        # Use only the version columns for trend calculation
        core_values = row[new_version_cols].values
        
        for i in range(1, len(core_values)):
            current = core_values[i]
            previous = core_values[i-1]
            
            if pd.notna(current):
                # Logic for Inflow (Previous was NaN, Current is not)
                if pd.isna(previous):
                    styles[i] = 'background-color: #d1ecf1; color: #0c5460;' # Light Blue (Inflow/New entry)
                # Logic for Increase
                elif current > previous:
                    styles[i] = 'background-color: #d4edda; color: #155724;' # Green (Increase)
                # Logic for Decrease
                elif current < previous:
                    styles[i] = 'background-color: #f8d7da; color: #721c24;' # Red (Decrease)
                # Logic for Stable
                else:
                    styles[i] = 'background-color: #e9ecef; color: #495057;' # Gray (Stable/No change)
        return styles

    # 5. Apply style and format
    # We specify subset=new_version_cols so Presence column doesn't get trend colors
    styled_df = df_wide.style.apply(style_trends, axis=1).format(
        precision=4, 
        na_rep=""
    )
    
    # 6. Render directly to Streamlit
    st.dataframe(styled_df, width='stretch')

def render_core_score_trajectory(df_core_score_full: pd.DataFrame, sorted_versions: list):
    """
    Renders a line chart of core scores over time.
    Defaults to the Top 5 developers from the latest version.
    Handles 'transient' gaps to prevent misleading straight lines during absence.
    """
    st.subheader("Developer Core Score Trajectory")

    # 1. Identify Top 5 from the latest version as default
    all_versions_clean = [v.replace('_', '.') for v in sorted_versions]
    latest_version = all_versions_clean[-1]
    
    df_latest = df_core_score_full[df_core_score_full['Version'] == latest_version]
    default_selection = df_latest.nlargest(5, 'Rombach_Core_Score')['Developer'].tolist()

    all_developers = sorted(df_core_score_full['Developer'].unique())
    selected_developers = st.multiselect(
        "Select Developers to Compare (Max 10):",
        options=all_developers,
        default=default_selection
    )
    st.markdown("\n")

    # 2. Constraints and validations
    if len(selected_developers) > 10:
        st.error("Please limit selection to 10 developers for better readability.")
        return
    
    if not selected_developers:
        st.info("Please select a developer to view their core_score trajectory.")
        return

    # 3. Data Reindexing to handle "Transient" gaps
    # Create a template of all versions to ensure NaNs exist where devs are inactive
    all_versions = [v.replace('_', '.') for v in sorted_versions]
    template = pd.MultiIndex.from_product(
        [selected_developers, all_versions], 
        names=['Developer', 'Version']
    ).to_frame(index=False)

    # Left merge ensures the line breaks when a developer is absent
    df_filtered = pd.merge(template, df_core_score_full, on=['Developer', 'Version'], how='left')

    # 4. Create Altair Chart
    # Use :N (Nominal) for Version to keep chronological order provided by sorted_versions
    chart = alt.Chart(df_filtered).mark_line(point=True).encode(
        x=alt.X('Version:N', title='Version', sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Rombach_Core_Score:Q', title='Core Score'),
        color=alt.Color('Developer:N', title='Developer'),
        tooltip=['Developer', 'Version', 'Rombach_Core_Score']
    ).properties(
        width='container',
        height=450
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
    st.caption("""
    Note: Breaks in the line indicate that the developer did not contribute to that specific version.
    """)


def render_risk_analysis_section(all_graphs: Dict[str, nx.Graph], sorted_versions: list):
    """
    Simulates the removal of the top developer (highest core score) for each version
    to examine structural dependency in the collaboration network.
    """

    risk_metrics = []

    for version in sorted_versions:
        # 1. Prepare Graph Copy
        G_original = all_graphs[version].copy()
        total_nodes = G_original.number_of_nodes()
        
        if total_nodes < 2:
            continue

        # 2. Get Initial State (Baseline)
        # Look at the largest connected component before removal
        initial_components = sorted(nx.connected_components(G_original), key=len, reverse=True)
        initial_lcc_size = len(initial_components[0])

        # 3. Identify and Remove Top Developer
        # Use the 'rombach_core_score' attribute
        core_score_data = nx.get_node_attributes(G_original, 'rombach_core_score')
        if not core_score_data:
            continue
            
        top_developer = max(core_score_data, key=core_score_data.get)
        G_simulated = G_original.copy()
        G_simulated.remove_node(top_developer)

        # 4. Calculate Impact
        # New LCC size after removal
        new_components = sorted(nx.connected_components(G_simulated), key=len, reverse=True)
        new_lcc_size = len(new_components[0]) if new_components else 0
        
        # Connectivity Loss: % reduction in LCC size
        # Formula: (Initial LCC - New LCC) / Initial LCC
        connectivity_loss = ((initial_lcc_size - new_lcc_size) / initial_lcc_size) * 100
        
        # Fragmented nodes -> difference in nodes no longer in the main component
        fragmented_nodes = initial_lcc_size - new_lcc_size - 1 # -1 is the removed node

        risk_metrics.append({
            'Version': version.replace('_', '.'),
            'Top_Developer': top_developer,
            'Connectivity_Loss': round(connectivity_loss, 2),
            'Fragmented_Nodes': fragmented_nodes
        })

    df_risk = pd.DataFrame(risk_metrics)

    if df_risk.empty:
        st.warning("Not enough data to perform risk simulation.")
        return

    # --- VISUAL 1: Snapshot for Selected Version ---
    st.subheader("Single Release Structural Dependency Snapshot")
    selected_ver = st.selectbox("Select Version to Simulate:", df_risk['Version'].tolist(), index=len(df_risk)-1)
    
    ver_data = df_risk[df_risk['Version'] == selected_ver].iloc[0]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.metric("Top Developer Removed", ver_data['Top_Developer'])
    col2.metric("Connectivity Loss", f"{ver_data['Connectivity_Loss']}%")
    col3.metric("Nodes Fragmented", int(ver_data['Fragmented_Nodes']))

    # --- VISUAL 2: Temporal Risk Trend ---
    st.subheader("Structural Dependency across Release Versions")
    
    risk_color = '#d97706' # Dark Amber 
    
    line_chart = alt.Chart(df_risk).mark_line(
        point=alt.OverlayMarkDef(color=risk_color, filled=True), 
        color=risk_color
    ).encode(
        x=alt.X('Version:N', sort=None, title="Version", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Connectivity_Loss:Q', title="Connectivity Loss (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=['Version', 'Top_Developer', 'Connectivity_Loss']
    ).properties(
        height=400,
        title="Connectivity Loss after Core Developer Removal"
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)
    st.caption("The line shows the percentage reduction in the largest connected component after the removal of the highest core-score developer.")




# MAIN FUNCTION
def main():
    st.title("Core–Periphery Structure in Developer Collaboration Networks")
    st.markdown("""
    This application analyzes core–periphery structures in an open-source software developer collaboration network across multiple release versions. By applying continuous core score measures, the app examines how developers’ structural positions vary between network snapshots.
    """
    )


    # SIDEBAR
    # Sidebar for Data Selection Mode
    st.sidebar.header("Data Configuration")
    data_mode = st.sidebar.radio(
        "Select Data Source:",
        options=["Use Example Data", "Upload Your Own Data"],
        index=0,
        help="Choose whether to use the pre-loaded data in the 'data/' folder or upload new CSV files."
    )

    all_dataframes = {}

    if data_mode == "Upload Your Own Data":
        uploaded_files = st.sidebar.file_uploader(
            "Upload Edge List Files (CSV)", 
            type="csv", 
            accept_multiple_files=True
        )
        if uploaded_files:
            all_dataframes = process_uploaded_files(uploaded_files)
            if all_dataframes:
                st.sidebar.success(f"Successfully loaded {len(all_dataframes)} file(s).")
                # Trigger re-calculation if data changes
                if st.sidebar.button("Process Uploaded Data"):
                    st.session_state.clear()
                    st.rerun()
        else:
            st.info("Please upload CSV files to proceed with custom analysis.")
            st.stop()
            
    else:
        # Use Example Data from Local Directory
        all_dataframes = load_all_network_data(DATA_DIR)
        if not all_dataframes:
            st.error("Example data not found in 'data/' directory.")
            st.stop()
        st.sidebar.info(f"Currently using {len(all_dataframes)} example snapshots.")

    # 2. Process Graphs and core_score Analysis
    # We use session state to ensure heavy calculations only run once per data set.
    if 'all_graphs' not in st.session_state:
         with st.spinner("Calculating Rombach core_score across all versions..."):
             st.session_state['all_graphs'], st.session_state['alg'] = process_all_graphs(all_dataframes)
         
    all_graphs = st.session_state['all_graphs']
    alg = st.session_state['alg']

    # sorted_version -> List : ["1_0","1_1_0"]
    sorted_versions = sorted(all_graphs.keys(), key=lambda x: [int(c) if c.isdigit() else c for c in x.split('_')])

    # 3. Get Data to visualize
    metrics_list = []   # 3A
    all_core_score_data_list = [] #3B
    

    for version in sorted_versions:
        G = all_graphs[version]
        
        # 3A. Network Size
        num_developers = G.number_of_nodes()
        num_edges = G.number_of_edges()

        metrics_list.append({
            'Version': version.replace('_', '.'), 
            'Num_Developers': num_developers,
            'Num_Edges': num_edges,
            'Snapshot_Key': version 
        })

        # 3B. core_score
        core_score_data = nx.get_node_attributes(G, 'rombach_core_score')

        if core_score_data:
            df_temp = pd.DataFrame(
                list(core_score_data.items()),
                columns=['Developer', 'Rombach_Core_Score']
            )
            df_temp['Rombach_Core_Score'] = df_temp['Rombach_Core_Score'].round(4)
            df_temp['Version'] = version.replace('_', '.')
            all_core_score_data_list.append(df_temp)

  

    # 3A. Network Size  
    df_overview = pd.DataFrame(metrics_list)
    if df_overview.empty:
        st.warning("No valid metrics calculated from the loaded data.")
        return
    
    # 3B. core_score
    df_core_score_full = pd.concat(all_core_score_data_list, ignore_index=True) if all_core_score_data_list else pd.DataFrame()


    # 4. Display Visualizations
    # 4A. OVERVIEW
    st.header("Overview")
    render_network_size_section(df_overview)

    # 4B. TOP core_score
    st.header("Comparative Core Score Analysis")
    render_top_developers_comparison(df_core_score_full, sorted_versions)

    # 4C. TEMPORAL TRAJECTORY
    st.header("Developer Dynamics")
    render_developer_profile_table(df_core_score_full, sorted_versions)
    render_core_score_trajectory(df_core_score_full, sorted_versions)

    # 4C. RISK ANALYSIS
    st.header("Dependency & Risk Analysis")
    st.markdown("""
    This analysis examines structural dependency in the collaboration network by simulating the removal of the developer with the highest Rombach core score in each version. It assesses the extent to which the main collaboration structure depends on a small number of core contributors.
    """)
    render_risk_analysis_section(all_graphs, sorted_versions)
    render_network_visualization(all_graphs, sorted_versions)



if __name__ == "__main__":
    main()