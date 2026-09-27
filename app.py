"""
Queueing Theory Decision Support System
A Streamlit-based web application for queueing analysis using Kendall notation.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from queueing_models import QueueingModel, QueueingResult

# Page configuration
st.set_page_config(
    page_title="Queueing Theory DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(94, 129, 244, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #5e81f4;
        font-size: 28px;
        font-weight: bold;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #5e81f4 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(94, 129, 244, 0.1);
        border: 1px solid rgba(94, 129, 244, 0.3);
        border-radius: 8px;
    }
    
    /* Tables */
    .dataframe {
        background: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Queueing Theory DSS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Decision Support System for Kendall Notation Queueing Models</p>', unsafe_allow_html=True)

# Sidebar - Model Selection and Parameters
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    
    # Model descriptions
    model_info = {
        "M/M/1:GD/∞/∞": "Single server, infinite queue capacity, infinite population",
        "M/M/1:GD/N/∞": "Single server, finite capacity N, infinite population",
        "M/M/c:GD/∞/∞": "Multiple servers (c), infinite queue capacity",
        "M/M/c:GD/N/∞": "Multiple servers (c), finite capacity N",
        "M/M/∞:GD/∞/∞": "Infinite servers (self-service model)"
    }
    
    model = st.selectbox(
        "🔷 Select Queueing Model",
        options=list(model_info.keys()),
        help="Choose a Kendall notation queueing model"
    )
    
    st.info(f"**Model:** {model_info[model]}")
    
    st.markdown("---")
    st.markdown("### 📥 Input Parameters")
    
    # Common parameters
    lambda_rate = st.number_input(
        "λ (Arrival Rate)",
        min_value=0.01,
        value=2.0,
        step=0.1,
        help="Average customers arriving per time unit"
    )
    
    mu_rate = st.number_input(
        "μ (Service Rate)",
        min_value=0.01,
        value=3.0,
        step=0.1,
        help="Average customers served per time unit per server"
    )
    
    # Conditional parameters
    c = 1  # Default
    N = None
    
    if "c" in model and "∞" not in model.split(":")[0].split("/")[2]:
        c = st.number_input(
            "c (Number of Servers)",
            min_value=1,
            value=2,
            step=1,
            help="Number of parallel servers"
        )
    elif "M/M/∞" in model:
        c = float('inf')
        st.info("∞ servers (self-service)")
    
    if "/N/" in model:
        N = st.number_input(
            "N (System Capacity)",
            min_value=1,
            value=10,
            step=1,
            help="Maximum customers allowed in the system"
        )
    
    max_n_display = st.slider(
        "Max n for P(n) display",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of probability states to display"
    )
    
    st.markdown("---")
    auto_calculate = st.checkbox("🔄 Auto-calculate", value=True, help="Automatically update results when parameters change")

# Main content - Always calculate and show results
if True: 
    # Create model and calculate
    qm = QueueingModel(lambda_rate, mu_rate)
    
    # Select appropriate method based on model
    if model == "M/M/1:GD/∞/∞":
        result = qm.mm1_infinite(max_n_display)
    elif model == "M/M/1:GD/N/∞":
        result = qm.mm1_finite(N, max_n_display)
    elif model == "M/M/c:GD/∞/∞":
        result = qm.mmc_infinite(c, max_n_display)
    elif model == "M/M/c:GD/N/∞":
        result = qm.mmc_finite(c, N, max_n_display)
    elif model == "M/M/∞:GD/∞/∞":
        result = qm.mm_infinity(max_n_display)
    
    # Check stability
    if not result.is_stable:
        st.error(f"⚠️ {result.error_message}")
        st.stop()
    
    # Display results
    st.markdown("## 📈 Performance Metrics")
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lₛ (Avg in System)", f"{result.Ls:.4f}")
        st.metric("Lq (Avg in Queue)", f"{result.Lq:.4f}")
    
    with col2:
        st.metric("Wₛ (Avg Time in System)", f"{result.Ws:.4f}")
        st.metric("Wq (Avg Time in Queue)", f"{result.Wq:.4f}")
    
    with col3:
        st.metric("P₀ (Idle Probability)", f"{result.P0:.4f}")
        st.metric("C̄ (Avg Being Served)", f"{result.C_bar:.4f}")
    
    with col4:
        st.metric("λₑff (Effective λ)", f"{result.lambda_eff:.4f}")
        if result.Pn is not None:
            st.metric("Pₙ (Blocking Prob)", f"{result.Pn:.4f}")
        else:
            st.metric("λ_lost (Lost Rate)", f"{result.lambda_lost:.4f}")
    
    # Utilization
    if c != float('inf') and c > 0:
        rho = lambda_rate / (c * mu_rate) if "c" in model else lambda_rate / mu_rate
        st.markdown("---")
        col_util1, col_util2, col_util3 = st.columns([1, 2, 1])
        with col_util2:
            st.markdown("### 🔄 Server Utilization (ρ)")
            progress_val = min(rho, 1.0)
            st.progress(progress_val)
            st.markdown(f"<h2 style='text-align: center; color: {'#ff4444' if rho > 0.9 else '#5e81f4'};'>{rho:.2%}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability Distribution
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        st.markdown("### 📊 Probability Distribution P(n)")
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(result.probabilities))),
            y=result.probabilities,
            marker=dict(
                color=result.probabilities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="P(n)")
            ),
            hovertemplate="n=%{x}<br>P(n)=%{y:.6f}<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis_title="Number of Customers (n)",
            yaxis_title="Probability P(n)",
            template="plotly_dark",
            height=400,
            margin=dict(l=50, r=50, t=30, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_table:
        st.markdown("### 📋 P(n) Values")
        
        # Create dataframe for display
        prob_data = {
            "n": list(range(len(result.probabilities))),
            "P(n)": [f"{p:.6f}" for p in result.probabilities]
        }
        
        st.dataframe(
            prob_data,
            hide_index=True,
            height=400,
            use_container_width=True
        )
        
        # Sum check
        total_prob = sum(result.probabilities)
        st.info(f"∑P(n) = {total_prob:.6f}")
    
    # Additional insights
    st.markdown("---")
    st.markdown("### 💡 Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown(f"""
        **Model Summary:**
        - **Model Type:** {result.model_name}
        - **Arrival Rate (λ):** {lambda_rate} customers/time unit
        - **Service Rate (μ):** {mu_rate} customers/time unit/server
        - **Traffic Intensity:** {lambda_rate/mu_rate:.4f}
        """)
    
    with insights_col2:
        if result.Lq > 0.1:
            st.warning("⚠️ Queue is building up. Consider adding capacity.")
        elif result.Lq < 0.01:
            st.success("✅ Queue is nearly empty. System is well-dimensioned.")
        
        if result.P0 > 0.5:
            st.info("💤 Server is idle more than 50% of the time.")
        elif result.P0 < 0.1:
            st.warning("⚡ Server is almost always busy.")
        
        if result.Pn is not None and result.Pn > 0.05:
            st.error(f"🚫 {result.Pn:.1%} of customers are being blocked!")

    # Understanding Kendall Notation - moved to expander below results
    with st.expander("📚 Understanding Kendall Notation"):
        st.markdown("""
        **Kendall notation** describes queueing systems using the format: **A/S/c:D/K/N**
        
        | Symbol | Meaning | Common Values |
        |--------|---------|---------------|
        | A | Arrival process | M (Markovian/Poisson) |
        | S | Service time distribution | M (Markovian/Exponential) |
        | c | Number of servers | 1, c (multiple), ∞ |
        | D | Queue discipline | GD (General Discipline/FIFO) |
        | K | System capacity | N (finite), ∞ (infinite) |
        | N | Calling population | ∞ (infinite) |
        
        **Performance Metrics:**
        - **Lₛ**: Expected number of customers in the system
        - **Lq**: Expected number of customers waiting in queue
        - **Wₛ**: Expected time a customer spends in the system
        - **Wq**: Expected time a customer spends waiting in queue
        - **P₀**: Probability that the system is empty (idle)
        - **ρ**: Server utilization (fraction of time server is busy)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Queueing Theory DSS | Built with Streamlit</p>",
    unsafe_allow_html=True
)
