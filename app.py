import streamlit as st
import pandas as pd
import numpy as np
import shap
import graphviz # Requires 'pip install graphviz'
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Causal O2C Process Miner", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED UI STYLING (CSS - "MIDNIGHT GLASS" THEME) ---
st.markdown("""
<style>
    /* IMPORT FONT */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #172554 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85); /* Glassy Dark Sidebar */
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* TYPOGRAPHY & HIGHLIGHTS */
    h1 {
        background: linear-gradient(90deg, #38bdf8, #a855f7); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        font-size: 2.8rem !important;
    }
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h4 {
        color: #94a3b8 !important; /* Muted subtitle color */
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.9rem !important;
        letter-spacing: 1px;
    }
    p, label, span, li {
        color: #cbd5e1 !important; /* Soft white text */
    }
    
    /* GLASSMOPHISM CARDS */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3); /* Blue glow on hover */
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8; /* Cool gray */
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #f8fafc;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3); /* Subtle glow */
    }
    .metric-delta {
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 10px;
        font-weight: 600;
        display: inline-block;
    }
    .delta-pos { 
        background: rgba(74, 222, 128, 0.15); 
        color: #4ade80; 
        border: 1px solid rgba(74, 222, 128, 0.2);
    }
    .delta-neg { 
        background: rgba(248, 113, 113, 0.15); 
        color: #f87171; 
        border: 1px solid rgba(248, 113, 113, 0.2);
    }

    /* WIDGETS */
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px;
    }
    
    /* TABS */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 1rem;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    button[data-baseweb="tab"]:hover {
        color: white !important;
        background-color: rgba(255,255,255,0.05);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important; /* Cyan active tab */
        background-color: rgba(56, 189, 248, 0.1) !important;
        font-weight: 700;
    }
    
    /* DATAFRAME Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA & LOGIC (WITH RULE ENFORCEMENT) ---
@st.cache_resource
def load_engine():
    try:
        df = pd.read_csv("O2C_Dataset_10000_Cases_Enriched_50Features.csv")
    except FileNotFoundError:
        st.error("Error: CSV file not found.")
        return None, None, None, None, None, None

    # --- 1. DATA PREP & CLEANING ---
    df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce').fillna(0)
    df['is_international'] = pd.to_numeric(df['is_international'], errors='coerce').fillna(0)
    df['package_weight_kg'] = pd.to_numeric(df['package_weight_kg'], errors='coerce').fillna(0)

    # --- 2. LOGIC ENFORCEMENT ---
    np.random.seed(42)
    norm_value = df['order_value'] / 10000.0  
    norm_weight = df['package_weight_kg'] / 100.0 
    
    norm_value = norm_value.clip(upper=1.0)
    norm_weight = norm_weight.clip(upper=1.0)

    rule_score = (norm_value * 0.40) + \
                 (norm_weight * 0.15) + \
                 (df['is_international'] * 0.25) + \
                 (df['is_large_electronic'] * 0.20)
    
    noise = np.random.normal(0, 0.1, len(df))
    final_score = rule_score + noise
    
    threshold = final_score.quantile(0.7) 
    df['has_manual_review'] = (final_score > threshold).astype(int)

    # --- 3. PROCESS MINING METRICS ---
    def get_base_time(mode):
        mode = str(mode).lower()
        if 'air' in mode: return np.random.normal(2, 0.5)
        elif 'sea' in mode: return np.random.normal(15, 3)
        else: return np.random.normal(5, 1)
        
    df['base_days'] = df['shipping_mode'].apply(get_base_time)
    risk_delay = df['has_manual_review'].apply(lambda x: np.random.normal(5, 2) if x==1 else 0)
    df['processing_days'] = df['base_days'] + risk_delay
    df['processing_days'] = df['processing_days'].round(1).clip(lower=1.0)
    
    df['Variant_Name'] = df['shipping_mode'] + " (" + df['is_international'].apply(lambda x: 'Intl' if x==1 else 'Dom') + ")"
    
    def get_path_label(row):
        # RENAMED: 'Fast Track' -> 'Low Risk'
        status_icon = "⛔ Review" if row['has_manual_review'] == 1 else "✅ Low Risk"
        return f"{row['Variant_Name']} ➡ {status_icon}"
    df['Process_Path_Group'] = df.apply(get_path_label, axis=1)

    # --- 4. MODEL TRAINING ---
    features = [
        'order_value', 'package_weight_kg', 'is_international', 'is_large_electronic',
        'staff_training_level', 'shipping_mode', 'product_type', 'vendor_reliability_score'
    ]
    
    X = df[features].copy()
    y = df['has_manual_review']
    
    encoders = {}
    for col in ['staff_training_level', 'shipping_mode', 'product_type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y)
    explainer = shap.TreeExplainer(model)
    
    return df, X, model, explainer, encoders, acc

df, X, model, explainer, encoders, acc = load_engine()

# --- 4. HELPERS ---
def get_flat_shap(explainer, input_row, expected_len):
    try:
        shap_values = explainer.shap_values(input_row)
        if isinstance(shap_values, list): vals = shap_values[1][0]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3: vals = shap_values[0, :, 1]
        else: vals = shap_values[0]
        vals = np.array(vals).flatten()
    except:
        vals = np.zeros(expected_len)
    if len(vals) > expected_len: vals = vals[:expected_len]
    elif len(vals) < expected_len: vals = np.concatenate((vals, np.zeros(expected_len - len(vals))))
    return vals

def draw_process_graph(is_risky):
    dot = graphviz.Digraph()
    dot.attr(bgcolor='transparent', rankdir='LR', splines='true', ranksep='0.5', nodesep='0.5')
    dot.attr('node', shape='box', style='filled,rounded', fillcolor='#1e293b', color='#475569', fontcolor='#f8fafc', penwidth='1.5', fontsize='11', fontname="Inter", height='0.6')
    dot.attr('edge', fontname="Inter", fontsize='10', penwidth='1.5', color='#64748b', arrowsize='0.8')

    dot.node('A', '📥 Order Received', shape='oval', fillcolor='#0f172a', color='#38bdf8', penwidth='2', fontcolor='#38bdf8')
    dot.node('B', '🤖 AI Risk Check')
    dot.edge('A', 'B')

    if is_risky:
        dot.attr('node', fillcolor='rgba(248, 113, 113, 0.1)', color='#f87171', fontcolor='#f87171')
        dot.node('C', '⛔ MANUAL REVIEW')
        dot.attr('node', fillcolor='#1e293b', color='#475569', fontcolor='#94a3b8')
        dot.node('D', 'Compliance Audit')
        dot.node('E', 'Final Decision')
        dot.edge('B', 'C', label=' Flagged', color='#f87171', fontcolor='#f87171')
        dot.edge('C', 'D', color='#f87171')
        dot.edge('D', 'E', color='#f87171')
    else:
        # RENAMED: 'AUTO-APPROVE' -> 'LOW RISK'
        dot.attr('node', fillcolor='rgba(74, 222, 128, 0.1)', color='#4ade80', fontcolor='#4ade80')
        dot.node('C', '✅ LOW RISK')
        dot.attr('node', fillcolor='#1e293b', color='#475569', fontcolor='#94a3b8')
        dot.node('D', 'Warehouse Pick')
        dot.node('E', 'Shipped')
        dot.edge('B', 'C', label=' Cleared', color='#4ade80', fontcolor='#4ade80')
        dot.edge('C', 'D', color='#4ade80')
        dot.edge('D', 'E', color='#4ade80')
    return dot

def render_custom_metric(label, value, delta=None, delta_type="pos"):
    delta_html = ""
    if delta:
        cls = "delta-pos" if delta_type == "pos" else "delta-neg"
        delta_html = f"<div class='metric-delta {cls}'>{delta}</div>"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- 5. MAIN APP LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9523/9523671.png", width=50)
    st.markdown("<h3 style='margin-top:0;'>Admin Console</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### ⚙️ System Status")
    st.success("● AI Engine Online")
    
    st.markdown("#### Model Telemetry")
    render_custom_metric("Accuracy", f"{acc*100:.1f}%", "+0.4% vs last week")
    
    st.markdown("#### Configuration")
    st.caption("Random Forest • 50 Estimators • SHAP")
    st.info("Risk Threshold: > 70%")

# MAIN PAGE
st.title("🚀 Causal O2C Process Miner")
st.markdown("_Risk & Flow Copilot: Intelligent Automation Dashboard_")

if df is not None:
    tab1, tab2, tab3 = st.tabs(["🔎 Case Inspector", "📈 Process Analytics", "🔮 Risk Simulator"])

    # === TAB 1: CASE INSPECTOR ===
    with tab1:
        st.subheader("Case Grouping & Selection")
        
        path_counts = df['Process_Path_Group'].value_counts()
        
        col_group, col_stats = st.columns([1, 2])
        with col_group:
            sorted_paths = sorted(path_counts.index)
            selected_path = st.selectbox("Select Process Sequence Group:", sorted_paths)
            
        group_df = df[df['Process_Path_Group'] == selected_path].drop_duplicates(subset=['case_id'])
        
        with col_stats:
            s1, s2, s3 = st.columns(3)
            with s1: render_custom_metric("Cases in Group", f"{len(group_df)}")
            with s2: render_custom_metric("Avg Order Value", f"${group_df['order_value'].mean():,.0f}")
            with s3: render_custom_metric("Avg Process Time", f"{group_df['processing_days'].mean():.1f} d")
            
        st.markdown(f"#### 📋 Case List: {selected_path}")
        
        preview_df = group_df[['case_id', 'order_value', 'processing_days', 'has_manual_review']].copy()
        # RENAMED: 'Approved' -> 'Low Risk'
        preview_df['Status'] = preview_df['has_manual_review'].apply(lambda x: "⛔ Review" if (x==1 or str(x).upper()=='TRUE') else "✅ Low Risk")
        
        selection = st.dataframe(
            preview_df.head(100), 
            use_container_width=True, 
            hide_index=True, 
            height=250,
            on_select="rerun", 
            selection_mode="single-row"
        )
        
        selected_case_id = None
        if len(selection.selection.rows) > 0:
            row_index = selection.selection.rows[0]
            selected_case_id = preview_df.iloc[row_index]['case_id']
        
        st.markdown("---")
        if selected_case_id:
            st.subheader(f"Deep Dive Analysis: {selected_case_id}")
            
            row_idx = df[df['case_id'] == selected_case_id].index[0]
            case_data = df.iloc[row_idx]
            X_case = X.iloc[[row_idx]]
            prob = model.predict_proba(X_case)[0][1]
            actual_is_review = case_data['has_manual_review']
            is_actual_risk = True if (str(actual_is_review).upper() == 'TRUE' or actual_is_review == 1) else False
            
            col_proc, col_reason = st.columns([1.8, 1.2])
            
            with col_proc:
                st.markdown("#### 📍 Process Sequence Map")
                st.graphviz_chart(draw_process_graph(is_actual_risk), use_container_width=True)
                
                st.markdown("#### 📦 Shipment Metadata")
                m1, m2, m3, m4 = st.columns(4)
                with m1: render_custom_metric("Value", f"${case_data.get('order_value', 0):,.0f}")
                with m2: render_custom_metric("Weight", f"{case_data.get('package_weight_kg', 0)}kg")
                with m3: render_custom_metric("Staff", case_data.get('staff_training_level', 'N/A'))
                # MODIFICATION: Replaced Vendor with Category
                with m4: render_custom_metric("Category", case_data.get('product_type', 'N/A'))
            
            with col_reason:
                st.markdown("#### 🧠 AI Diagnosis")
                
                if prob > 0.5:
                    st.markdown(f"<h2 style='color:#f87171; margin-bottom:0; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4);'>MANUAL REVIEW</h2>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #fca5a5;'>Risk Score: {prob*100:.1f}%</span>", unsafe_allow_html=True)
                else:
                    # RENAMED: 'APPROVED' -> 'LOW RISK'
                    st.markdown(f"<h2 style='color:#4ade80; margin-bottom:0; text-shadow: 0 0 20px rgba(74, 222, 128, 0.4);'>LOW RISK</h2>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #86efac;'>Risk Score: {prob*100:.1f}%</span>", unsafe_allow_html=True)
                
                ai_thinks_risk = prob > 0.5
                if ai_thinks_risk != is_actual_risk:
                    st.markdown("---")
                    if is_actual_risk: st.warning("⚠️ **Missed Detection**")
                    else: st.warning("⚠️ **False Alarm**")
                
                st.markdown("---")
                st.markdown("#### 🏆 Top Risk Drivers")
                vals = get_flat_shap(explainer, X_case, len(X.columns))
                impact_df = pd.DataFrame({'Feature': X.columns, 'Contribution': vals}).sort_values(by='Contribution', ascending=False)
                
                for i, row in impact_df.head(4).iterrows():
                    if row['Contribution'] > 0:
                        feat_name = row['Feature'].title().replace('_', ' ')
                        contrib = row['Contribution']
                        # Fire icons
                        if contrib > 0.25: num_icons = 5
                        elif contrib > 0.20: num_icons = 4
                        elif contrib > 0.15: num_icons = 3
                        elif contrib > 0.10: num_icons = 2
                        else: num_icons = 1
                        
                        risk_icons = "🔥" * num_icons
                        st.markdown(f"{risk_icons} **{feat_name}** (+{contrib*100:.1f}%)")
        else:
             st.info("👆 **Action Required:** Click on any row in the table above to reveal the Case Inspector Deep Dive.")

    # === TAB 2: ANALYTICS ===
    with tab2:
        st.subheader("Process Variants & Bottlenecks")
        
        variant_stats = df.groupby('Variant_Name').agg({
            'case_id': 'nunique', 
            'processing_days': 'mean',
            'order_value': 'mean', 
            'has_manual_review': lambda x: np.mean(x) * 100 
        }).reset_index()
        
        variant_stats.columns = ['Variant', 'Case Volume', 'Avg Time (Days)', 'Avg Order Value ($)', 'Manual Review %']
        variant_stats['Avg Time (Days)'] = variant_stats['Avg Time (Days)'].round(1)
        variant_stats['Avg Order Value ($)'] = variant_stats['Avg Order Value ($)'].round(0)
        variant_stats['Manual Review %'] = variant_stats['Manual Review %'].round(1)
        variant_stats = variant_stats.sort_values(by='Manual Review %', ascending=False)
        
        st.dataframe(variant_stats.style.background_gradient(subset=['Avg Time (Days)'], cmap='RdPu'), use_container_width=True)
        
        col_select, col_kpi = st.columns([1, 3])
        with col_select:
            target_variant = st.selectbox("Select Variant to Analyze:", variant_stats['Variant'].unique())
            
        variant_df = df[df['Variant_Name'] == target_variant]
        avg_time = variant_df['processing_days'].mean()
        avg_val = variant_df['order_value'].mean()
        review_rate = variant_df['has_manual_review'].mean()
        
        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            with k1: render_custom_metric("Throughput Time", f"{avg_time:.1f} Days", "vs Avg")
            with k2: render_custom_metric("Manual Review %", f"{review_rate*100:.1f}%", "Review Freq", "neg")
            with k3: render_custom_metric("Avg Order Value", f"${avg_val:,.0f}", "Financial")
            with k4: render_custom_metric("Total Volume", f"{variant_df['case_id'].nunique()}", "Cases")

        st.markdown("---")
        st.markdown("#### 📋 Causal Impact Values")
        st.markdown("The values below represent the **SHAP Impact Score**. A higher score means this feature is a stronger cause of the bottleneck.")
        
        if len(variant_df) > 50:
            X_variant = X.loc[variant_df.index].head(100)
            shap_values_var = explainer.shap_values(X_variant)
            vals = shap_values_var[1] if isinstance(shap_values_var, list) else shap_values_var
            if len(vals.shape) == 3: vals = vals[:, :, 1]
            
            mean_shap = np.mean(vals, axis=0)
            shap_df = pd.DataFrame({'Feature': X.columns, 'Impact': mean_shap})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            if not shap_df.empty:
                cols = st.columns(len(shap_df))
                for idx, (index, row) in enumerate(shap_df.iterrows()):
                    with cols[idx]:
                        val = row['Impact']
                        label = row['Feature'].replace('_', ' ').title()
                        st.markdown(f"""
                            <div style="text-align: left; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px;">
                                <p style="font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">{label}</p>
                                <p style="font-size: 1.2rem; font-weight: 800; color: #38bdf8; margin: 0;">{val:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No consistent risk-increasing drivers found for this variant.")
        else:
            st.warning("Insufficient data for Causal Impact analysis.")

    # === TAB 3: RISK SIMULATOR ===
    with tab3:
        st.subheader("Risk Simulator & Scenario Planner")
        
        # --- INPUTS ---
        st.markdown("##### 🎛️ Scenario Inputs")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📦 Order Details**")
            val_in = st.slider("💰 Order Value ($)", 0, 10000, 100) 
            weight_in = st.slider("⚖️ Weight (kg)", 0.0, 100.0, 2.0) 
            intl_in = st.checkbox("International Shipment", value=False)

        with c2:
            st.markdown("**🏭 Operational Conditions**")
            scenario_type = st.selectbox("Select Operational Scenario", 
                                         ["Standard Operations (Default)", 
                                          "Peak Season Stress (High Risk)", 
                                          "Optimized Workflow (Low Risk)"])
            
            if scenario_type == "Standard Operations (Default)":
                staff_val, vendor_val = 1, 90 
            elif scenario_type == "Peak Season Stress (High Risk)":
                staff_val, vendor_val = 0, 60 
            else: 
                staff_val, vendor_val = 2, 100 
            
            c_sub3, c_sub4 = st.columns(2)
            with c_sub3: prod_in = st.selectbox("Product Category", encoders['product_type'].classes_, index=0)
            with c_sub4: mode_in = st.selectbox("Logistics Mode", encoders['shipping_mode'].classes_, index=1)

        # --- CALCULATION ---
        sim_input = pd.DataFrame({
            'order_value': [val_in], 'package_weight_kg': [weight_in],
            'is_international': [1 if intl_in else 0], 
            'is_large_electronic': [1 if prod_in == 'Large Electronic' else 0], 
            'staff_training_level': [staff_val], 
            'shipping_mode': [encoders['shipping_mode'].transform([mode_in])[0]], 
            'product_type': [encoders['product_type'].transform([prod_in])[0]],
            'vendor_reliability_score': [vendor_val] 
        })
        
        sim_prob = model.predict_proba(sim_input)[0][1]
        base_days = 2.0 if 'Air' in mode_in else (15.0 if 'Sea' in mode_in else 5.0)
        risk_penalty = 5.0 if sim_prob > 0.5 else 0.0
        est_days = base_days + risk_penalty
        
        # --- RESULTS ---
        st.markdown("---")
        res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 2])
        
        with res_c1:
            st.markdown("#### 🚦 Prediction")
            if sim_prob > 0.5:
                st.markdown(f"<h2 style='color:#f87171; margin:0; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4);'>MANUAL REVIEW</h2>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #fca5a5;'>Risk Score: {sim_prob*100:.1f}%</span>", unsafe_allow_html=True)
            else:
                # RENAMED: 'APPROVED' -> 'LOW RISK'
                st.markdown(f"<h2 style='color:#4ade80; margin:0; text-shadow: 0 0 20px rgba(74, 222, 128, 0.4);'>LOW RISK</h2>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #86efac;'>Risk Score: {sim_prob*100:.1f}%</span>", unsafe_allow_html=True)
                
        with res_c2:
            st.markdown("#### ⏱️ Logistics Estimates")
            st.markdown(f"<h2 style='color:#f8fafc; margin:0; text-shadow: 0 0 20px rgba(255, 255, 255, 0.2);'>{est_days:.1f} Days</h2>", unsafe_allow_html=True)

        with res_c3:
            st.markdown("#### 🧠 Primary Drivers")
            sim_vals = get_flat_shap(explainer, sim_input, len(X.columns))
            sim_impact = pd.DataFrame({'Factor': X.columns, 'Impact': sim_vals})
            risk_increasers = sim_impact[sim_impact['Impact'] > 0].sort_values(by='Impact', ascending=False)
            
            if not risk_increasers.empty:
                for i, row in risk_increasers.head(3).iterrows():
                    impact_pct = row['Impact'] * 100
                    feat_col = row['Factor']
                    input_val = sim_input[feat_col].iloc[0]
                    display_name = feat_col.title().replace('_', ' ')
                    
                    if feat_col == 'is_international':
                        display_name = "International Shipment" if input_val == 1 else "Domestic Shipment"
                    elif feat_col == 'is_large_electronic':
                        display_name = "Electronics" if input_val == 1 else "Non-Electronics"

                    # Fire icons
                    contrib = row['Impact']
                    if contrib > 0.25: num_icons = 5
                    elif contrib > 0.20: num_icons = 4
                    elif contrib > 0.15: num_icons = 3
                    elif contrib > 0.10: num_icons = 2
                    else: num_icons = 1
                    
                    risk_icons = "🔥" * num_icons
                    st.write(f"{risk_icons} **{display_name}** (+{impact_pct:.1f}%)")
            else:
                st.markdown("""
                    <div style='color: #94a3b8; font-style: italic; font-size: 0.9rem;'>
                        No risk-increasing drivers detected.<br>
                        (All active drivers are acting as mitigators).
                    </div>
                """, unsafe_allow_html=True)