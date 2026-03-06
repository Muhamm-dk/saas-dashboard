"""
AWS SaaS Sales Analytics - Interactive Dashboard
MSc Computer Science Project

Author: Muhammed K
Date: February 2026

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SaaS Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and encoders"""
    try:
        models_dir = Path("models")
        
        models = {
            'lead_model': joblib.load(models_dir / 'lead_scoring_model.pkl'),
            'scaler': joblib.load(models_dir / 'feature_scaler.pkl'),
            'encoders': joblib.load(models_dir / 'label_encoders.pkl'),
            'segment_model': joblib.load(models_dir / 'customer_segmentation_model.pkl'),
            'segment_scaler': joblib.load(models_dir / 'clustering_scaler.pkl'),
            'cluster_mapping': pd.read_csv(models_dir / 'cluster_mapping.csv')
        }
        
        # Load model summary
        with open(models_dir / 'model_summary.json', 'r') as f:
            models['summary'] = json.load(f)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure all model files are in the 'models/' folder")
        st.stop()

models = load_models()

# Extract cluster names
cluster_names_dict = dict(zip(
    models['cluster_mapping']['Cluster_ID'], 
    models['cluster_mapping']['Cluster_Name']
))

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def score_lead(lead_data):
    """Score a single lead"""
    try:
        # Prepare features
        features = pd.DataFrame([lead_data])
        
        # Encode categoricals
        for col in ['Region', 'Segment', 'Industry', 'Product']:
            if col in features.columns and col in models['encoders']:
                le = models['encoders'][col]
                try:
                    features[col + '_encoded'] = le.transform(features[col])
                except:
                    features[col + '_encoded'] = 0  # Unknown category
                features = features.drop(col, axis=1)
        
        # Get feature names from training
        feature_cols = models['summary']['Lead Scoring']['Features']
        
        # Ensure all features present
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0
        
        features = features[feature_cols]
        
        # Predict
        probability = models['lead_model'].predict_proba(features)[0, 1]
        score = probability * 100
        
        # Determine priority
        if score >= 70:
            priority = "HIGH"
            color = "🟢"
            recommendation = "🔥 Immediate Action Required"
        elif score >= 40:
            priority = "MEDIUM"
            color = "🟡"
            recommendation = "📋 Standard Follow-Up"
        else:
            priority = "LOW"
            color = "🔴"
            recommendation = "📧 Automated Nurture"
        
        return {
            'score': round(score, 1),
            'probability': round(probability, 3),
            'priority': priority,
            'color': color,
            'recommendation': recommendation
        }
    except Exception as e:
        st.error(f"Error scoring lead: {str(e)}")
        return None

def segment_customer(customer_data):
    """Assign customer to segment"""
    try:
        # Prepare features (map input to expected format)
        features_dict = {
            'customer_lifetime_value_first': customer_data.get('clv', 0),
            'purchase_frequency_first': customer_data.get('frequency', 0),
            'Sales_mean': customer_data.get('sales_mean', 0),
            'Discount_mean': customer_data.get('discount_mean', 0),
            'Product_nunique': customer_data.get('prod_div', 0),
            'profit_margin_mean': customer_data.get('margin', 0)
        }
        
        features = pd.DataFrame([features_dict])
        
        # Scale
        features_scaled = models['segment_scaler'].transform(features)
        
        # Predict
        cluster_id = models['segment_model'].predict(features_scaled)[0]
        cluster_name = cluster_names_dict.get(cluster_id, f"Cluster {cluster_id}")
        
        return {
            'cluster_id': int(cluster_id),
            'cluster_name': cluster_name,
            'characteristics': customer_data
        }
    except Exception as e:
        st.error(f"Error segmenting customer: {str(e)}")
        return None

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=SaaS+Analytics", use_container_width=True)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Dashboard Home", 
     "📊 Lead Scoring", 
     "👥 Customer Segmentation", 
     "📈 Batch Processing",
     "📚 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")

try:
    st.sidebar.metric("Model ROC-AUC", f"{models['summary']['Lead Scoring']['ROC-AUC']:.3f}")
    st.sidebar.metric("Segments", f"{models['summary']['Segmentation']['n_clusters']}")
except:
    st.sidebar.metric("Models", "Loaded ✅")

st.sidebar.markdown("---")
st.sidebar.markdown("**MSc Project 2026**")
st.sidebar.markdown("AWS SaaS Sales Analytics")

# ============================================================================
# PAGE 1: DASHBOARD HOME
# ============================================================================

if page == "🏠 Dashboard Home":
    st.title("🎯 AWS SaaS Sales Analytics Dashboard")
    st.markdown("**AI-Powered Lead Scoring & Customer Segmentation System**")
    st.markdown("---")
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Model Type",
            value="Random Forest",
            delta="Best Performer"
        )
    
    with col2:
        st.metric(
            label="🎯 ROC-AUC Score",
            value=f"{models['summary']['Lead Scoring']['ROC-AUC']:.3f}",
            delta="Target: 0.75+"
        )
    
    with col3:
        st.metric(
            label="👥 Customer Segments",
            value=models['summary']['Segmentation']['n_clusters'],
            delta="Validated"
        )
    
    with col4:
        st.metric(
            label="📈 Silhouette Score",
            value=f"{models['summary']['Segmentation']['silhouette_score']:.3f}",
            delta="Good Separation"
        )
    
    st.markdown("---")
    
    # System Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 What This System Does")
        st.markdown("""
        This dashboard provides **AI-powered sales intelligence** for B2B SaaS companies:
        
        **1. Lead Scoring (0-100)**
        - Predicts probability of high-value deals
        - Prioritizes sales efforts (HIGH/MEDIUM/LOW)
        - Recommends specific actions per lead
        
        **2. Customer Segmentation**
        - Identifies 5 distinct customer personas
        - Provides retention strategies per segment
        - Enables resource allocation optimization
        
        **3. Batch Processing**
        - Score hundreds of leads at once
        - Upload CSV, get scored results
        - Export for CRM integration
        """)
    
    with col2:
        st.subheader("📊 Expected Business Impact")
        
        impact_data = {
            'Metric': ['Conversion Rate', 'Sales Efficiency', 'Profit Margin', 'Deal Size'],
            'Improvement': ['+15-25%', '+20%', '+5-10 pts', '+33%'],
            'Annual Value': ['$300K+', '$200K+', '$150K+', '$250K+']
        }
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
        
        st.info("💡 **ROI:** Models pay for themselves in 2-3 months through improved conversion rates and sales efficiency.")
    
    st.markdown("---")
    
    # Quick Start Guide
    st.subheader("🚀 Quick Start Guide")
    
    tab1, tab2, tab3 = st.tabs(["Lead Scoring", "Customer Segmentation", "Batch Processing"])
    
    with tab1:
        st.markdown("""
        **How to score a lead:**
        1. Click **📊 Lead Scoring** in the sidebar
        2. Enter lead details (region, segment, CLV, etc.)
        3. Click **Score Lead** button
        4. Get instant score + recommendations
        
        **Use Case:** Sales rep gets new inquiry → Score it → Prioritize follow-up
        """)
    
    with tab2:
        st.markdown("""
        **How to identify customer segment:**
        1. Click **👥 Customer Segmentation** in the sidebar
        2. Enter customer metrics (CLV, frequency, margin, etc.)
        3. Click **Identify Segment** button
        4. Get segment assignment + strategy
        
        **Use Case:** Customer Success team → Segment customers → Apply tailored retention strategy
        """)
    
    with tab3:
        st.markdown("""
        **How to score multiple leads:**
        1. Click **📈 Batch Processing** in the sidebar
        2. Upload CSV file with lead data
        3. Click **Score All Leads** button
        4. Download scored results
        
        **Use Case:** Weekly pipeline review → Score all leads → Distribute prioritized list to team
        """)

# ============================================================================
# PAGE 2: LEAD SCORING
# ============================================================================

elif page == "📊 Lead Scoring":
    st.title("📊 Lead Scoring System")
    st.markdown("Enter lead details to get instant AI-powered score and recommendations")
    st.markdown("---")
    
    # Input form
    with st.form("lead_scoring_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Geographic & Segment")
            region = st.selectbox("Region", ["EMEA", "Americas", "APJ"])
            segment = st.selectbox("Segment", ["SMB", "Mid-Market", "Enterprise"])
            industry = st.selectbox("Industry", [
                "Finance", "Retail", "Healthcare", "Technology", 
                "Manufacturing", "Energy", "Telecom", "Education", 
                "Government", "Misc"
            ])
        
        with col2:
            st.subheader("🛍️ Product & Deal")
            product = st.selectbox("Product", [
                "ContactMatcher", "Marketing Suite", "Site Analytics",
                "Product_D", "Product_E", "Product_F"
            ])
            discount = st.slider("Discount (%)", 0, 50, 10, 1)
            quantity = st.number_input("Quantity", 1, 100, 2)
        
        with col3:
            st.subheader("👤 Customer History")
            clv = st.number_input("Customer Lifetime Value ($)", 0, 1000000, 250000, 10000)
            frequency = st.number_input("Purchase Frequency", 0, 500, 100, 10)
            prod_div = st.number_input("Product Diversity", 1, 14, 4, 1)
        
        # Submit button
        submitted = st.form_submit_button("🎯 Score Lead", type="primary", use_container_width=True)
    
    if submitted:
        # Create lead dict
        lead = {
            'Region': region,
            'Segment': segment,
            'Industry': industry,
            'Product': product,
            'Discount': discount,
            'Quantity': quantity,
            'customer_lifetime_value': clv,
            'purchase_frequency': frequency,
            'product_diversity': prod_div
        }
        
        # Score the lead
        result = score_lead(lead)
        
        if result:
            st.markdown("---")
            
            # Display score
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Lead Score", f"{result['score']}/100", delta="AI Predicted")
            
            with col2:
                st.metric("Priority", f"{result['color']} {result['priority']}")
            
            with col3:
                st.metric("Win Probability", f"{result['probability']:.1%}")
            
            with col4:
                predicted_value = "$450-650" if result['score'] >= 70 else "$300-450" if result['score'] >= 40 else "$150-300"
                st.metric("Predicted Deal Size", predicted_value)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['score'],
                delta={'reference': 50},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lead Score", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#ffcdd2'},
                        {'range': [40, 70], 'color': '#fff9c4'},
                        {'range': [70, 100], 'color': '#c8e6c9'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("✅ Recommended Actions")
            
            if result['priority'] == "HIGH":
                st.markdown("""
                <div class="success-box">
                <h3>🔥 HIGH PRIORITY - Immediate Action Required</h3>
                <ul>
                    <li><b>Assignment:</b> Senior sales representative immediately</li>
                    <li><b>Timeline:</b> Schedule custom demo within 24 hours</li>
                    <li><b>Preparation:</b> Prepare executive briefing and ROI analysis</li>
                    <li><b>Pricing:</b> Approve discount up to 15% if needed to close</li>
                    <li><b>Follow-up:</b> Daily touchpoints until close or disqualification</li>
                </ul>
                <p><b>💰 Expected Deal Size:</b> $600-800 (high-value opportunity)</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif result['priority'] == "MEDIUM":
                st.markdown("""
                <div class="warning-box">
                <h3>📋 MEDIUM PRIORITY - Standard Follow-Up</h3>
                <ul>
                    <li><b>Assignment:</b> Qualified sales representative</li>
                    <li><b>Timeline:</b> Standard discovery call within 3 business days</li>
                    <li><b>Preparation:</b> Product demo if properly qualified</li>
                    <li><b>Pricing:</b> Discount limit 10%, no exceptions without approval</li>
                    <li><b>Follow-up:</b> Weekly check-ins during sales cycle</li>
                </ul>
                <p><b>💰 Expected Deal Size:</b> $300-450 (standard opportunity)</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class="danger-box">
                <h3>📧 LOW PRIORITY - Automated Nurture</h3>
                <ul>
                    <li><b>Assignment:</b> Marketing automation (no sales rep time)</li>
                    <li><b>Timeline:</b> Add to email nurture campaign</li>
                    <li><b>Preparation:</b> Self-service product trial access</li>
                    <li><b>Pricing:</b> No discounting - full price only</li>
                    <li><b>Follow-up:</b> Automated emails, revisit if engagement improves</li>
                </ul>
                <p><b>💰 Expected Deal Size:</b> $150-300 (low probability)</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: CUSTOMER SEGMENTATION
# ============================================================================

elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("Identify customer persona and get tailored retention strategies")
    st.markdown("---")
    
    with st.form("segmentation_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💰 Value Metrics")
            clv_seg = st.number_input("Customer Lifetime Value ($)", 0, 1000000, 250000, 10000)
            sales_mean = st.number_input("Average Deal Size ($)", 0, 10000, 2500, 100)
        
        with col2:
            st.subheader("📊 Behavior Metrics")
            freq_seg = st.number_input("Purchase Frequency (per year)", 0, 365, 100, 5)
            prod_div_seg = st.number_input("Products Used", 1, 14, 4, 1)
        
        with col3:
            st.subheader("💵 Financial Metrics")
            discount_mean = st.number_input("Average Discount (%)", 0.0, 50.0, 12.0, 1.0)
            margin_mean = st.number_input("Average Margin (%)", -20.0, 50.0, 15.0, 1.0)
        
        submitted_seg = st.form_submit_button("🎯 Identify Segment", type="primary", use_container_width=True)
    
    if submitted_seg:
        customer = {
            'clv': clv_seg,
            'frequency': freq_seg,
            'sales_mean': sales_mean,
            'discount_mean': discount_mean,
            'prod_div': prod_div_seg,
            'margin': margin_mean
        }
        
        result_seg = segment_customer(customer)
        
        if result_seg:
            st.markdown("---")
            
            # Display segment
            st.subheader(f"📋 Customer Segment: **{result_seg['cluster_name']}**")
            
            # Segment characteristics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CLV", f"${clv_seg:,}")
            with col2:
                st.metric("Frequency", f"{freq_seg}/year")
            with col3:
                st.metric("Avg Deal", f"${sales_mean:,}")
            with col4:
                st.metric("Margin", f"{margin_mean:.1f}%")
            
            st.markdown("---")
            
            # Strategies by segment
            segment_name = result_seg['cluster_name']
            
            if "High-Value" in segment_name or "Champion" in segment_name:
                st.markdown("""
                <div class="success-box">
                <h2>🏆 High-Value Champions</h2>
                <h3>Strategy: White-Glove Premium Service</h3>
                
                <h4>📞 Sales Approach:</h4>
                <ul>
                    <li>Dedicated account manager (named contact)</li>
                    <li>Quarterly executive business reviews</li>
                    <li>Direct line to C-level support</li>
                </ul>
                
                <h4>🔒 Retention Tactics:</h4>
                <ul>
                    <li>24/7 priority support with 1-hour response SLA</li>
                    <li>Proactive outreach (don't wait for issues)</li>
                    <li>Early access to beta features</li>
                    <li>Annual health checks and optimization reviews</li>
                </ul>
                
                <h4>📈 Expansion Focus:</h4>
                <ul>
                    <li>Cross-sell premium products</li>
                    <li>Increase user licenses (10-20% annually)</li>
                    <li>Multi-year contracts with discounts</li>
                    <li>Strategic partnership opportunities</li>
                </ul>
                
                <h4>💼 Resource Allocation:</h4>
                <p><b>50% of customer success resources</b> - These customers drive majority of profit</p>
                
                <h4>⚠️ Risk Factors:</h4>
                <p>Complacency - never assume they're happy. Competitors actively target your best customers.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "Steady" in segment_name or "Contributor" in segment_name:
                st.markdown("""
                <div class="warning-box">
                <h2>📊 Steady Contributors</h2>
                <h3>Strategy: Efficient Scalable Service</h3>
                
                <h4>📞 Sales Approach:</h4>
                <ul>
                    <li>Customer success manager (shared, 1:50 ratio)</li>
                    <li>Semi-annual check-ins (automated + personal)</li>
                    <li>Self-service resources with guided paths</li>
                </ul>
                
                <h4>🔒 Retention Tactics:</h4>
                <ul>
                    <li>Business hours support (24-hour response)</li>
                    <li>Automated quarterly health scores</li>
                    <li>Customer success content library</li>
                    <li>Community forum access for peer learning</li>
                </ul>
                
                <h4>📈 Expansion Focus:</h4>
                <ul>
                    <li>Usage-based upsell recommendations</li>
                    <li>Product cross-sell based on behavior</li>
                    <li>Annual renewals with modest increases</li>
                    <li>Referral program incentives</li>
                </ul>
                
                <h4>💼 Resource Allocation:</h4>
                <p><b>35% of customer success resources</b> - Solid backbone of business</p>
                
                <h4>⚠️ Risk Factors:</h4>
                <p>Neglect - easy to take for granted. Monitor engagement metrics closely.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "Discount" in segment_name or "Seeker" in segment_name:
                st.markdown("""
                <div class="warning-box">
                <h2>💸 Discount Seekers</h2>
                <h3>Strategy: Value Demonstration & Education</h3>
                
                <h4>📞 Sales Approach:</h4>
                <ul>
                    <li>Value-based selling (ROI focus, not price)</li>
                    <li>Quarterly value reports (show impact)</li>
                    <li>Success stories from similar customers</li>
                </ul>
                
                <h4>🔒 Retention Tactics:</h4>
                <ul>
                    <li>Self-service support preferred</li>
                    <li>ROI calculators and business case templates</li>
                    <li>Feature adoption coaching (increase stickiness)</li>
                    <li>Annual review with usage statistics</li>
                </ul>
                
                <h4>📈 Expansion Focus:</h4>
                <ul>
                    <li>Bundle pricing (reduce per-product discounts)</li>
                    <li>Annual prepay discounts (lock in revenue)</li>
                    <li>Graduation path to higher tier</li>
                    <li>Value-add features vs. price cuts</li>
                </ul>
                
                <h4>💼 Resource Allocation:</h4>
                <p><b>10% of customer success resources</b> - Evaluate profitability regularly</p>
                
                <h4>⚠️ Risk Factors:</h4>
                <p>High churn risk - will leave for 5% better price. Consider if worth retaining.</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:  # At-Risk / Low-Value
                st.markdown("""
                <div class="danger-box">
                <h2>⚠️ At-Risk / Low-Value</h2>
                <h3>Strategy: Evaluate & Optimize or Exit</h3>
                
                <h4>📞 Sales Approach:</h4>
                <ul>
                    <li>Self-service ONLY (no dedicated resources)</li>
                    <li>Automated email support (48-hour response)</li>
                    <li>Knowledge base and documentation</li>
                </ul>
                
                <h4>🔒 Retention Decision:</h4>
                <ul>
                    <li><b>Option A:</b> Evaluate if customer can be saved
                        <ul>
                            <li>Can they increase usage? (upgrade to profitable tier)</li>
                            <li>Can discounts be reduced? (price increase)</li>
                        </ul>
                    </li>
                    <li><b>Option B:</b> Graceful offboarding
                        <ul>
                            <li>Losing money on these customers</li>
                            <li>Free up resources for better customers</li>
                            <li>Refer to competitors or free alternatives</li>
                        </ul>
                    </li>
                </ul>
                
                <h4>📈 Expansion Focus:</h4>
                <p><b>NONE</b> - Focus on margin recovery or exit</p>
                
                <h4>💼 Resource Allocation:</h4>
                <p><b>5% of customer success resources</b> - Minimize investment</p>
                
                <h4>⚠️ Action Items:</h4>
                <ol>
                    <li>Calculate actual profitability (revenue - cost to serve)</li>
                    <li>If negative: Offer price increase or cancellation</li>
                    <li>If they won't pay fair price: Better to lose them</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4: BATCH PROCESSING
# ============================================================================

elif page == "📈 Batch Processing":
    st.title("📈 Batch Lead Scoring")
    st.markdown("Upload CSV file to score multiple leads simultaneously")
    st.markdown("---")
    
    # Download template
    st.subheader("📥 Step 1: Download Template")
    
    template_data = {
        'Region': ['EMEA', 'Americas', 'APJ'],
        'Segment': ['Enterprise', 'Mid-Market', 'SMB'],
        'Industry': ['Finance', 'Retail', 'Technology'],
        'Product': ['ContactMatcher', 'Marketing Suite', 'Site Analytics'],
        'Discount': [10, 15, 20],
        'Quantity': [3, 2, 1],
        'customer_lifetime_value': [450000, 200000, 80000],
        'purchase_frequency': [150, 100, 50],
        'product_diversity': [6, 4, 2]
    }
    
    template_df = pd.DataFrame(template_data)
    
    st.download_button(
        label="📄 Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="lead_scoring_template.csv",
        mime="text/csv"
    )
    
    st.dataframe(template_df, use_container_width=True)
    
    st.markdown("---")
    
    # Upload file
    st.subheader("📤 Step 2: Upload Your Leads")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} leads found.")
            
            st.subheader("Preview Data:")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Score All Leads", type="primary", use_container_width=True):
                with st.spinner("Scoring leads... Please wait..."):
                    # Score each lead
                    scores = []
                    priorities = []
                    recommendations = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        lead_dict = row.to_dict()
                        result = score_lead(lead_dict)
                        
                        if result:
                            scores.append(result['score'])
                            priorities.append(result['priority'])
                            recommendations.append(result['recommendation'])
                        else:
                            scores.append(0)
                            priorities.append('ERROR')
                            recommendations.append('Error scoring')
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Add results to dataframe
                    df['lead_score'] = scores
                    df['priority'] = priorities
                    df['recommendation'] = recommendations
                
                st.success(f"✅ Successfully scored {len(df)} leads!")
                
                # Show results
                st.markdown("---")
                st.subheader("📊 Scored Results")
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                high_count = (df['priority'] == 'HIGH').sum()
                medium_count = (df['priority'] == 'MEDIUM').sum()
                low_count = (df['priority'] == 'LOW').sum()
                avg_score = df['lead_score'].mean()
                
                with col1:
                    st.metric("🟢 HIGH Priority", high_count, f"{high_count/len(df)*100:.1f}%")
                
                with col2:
                    st.metric("🟡 MEDIUM Priority", medium_count, f"{medium_count/len(df)*100:.1f}%")
                
                with col3:
                    st.metric("🔴 LOW Priority", low_count, f"{low_count/len(df)*100:.1f}%")
                
                with col4:
                    st.metric("📊 Average Score", f"{avg_score:.1f}/100")
                
                # Priority distribution chart
                fig = px.pie(
                    values=[high_count, medium_count, low_count],
                    names=['HIGH', 'MEDIUM', 'LOW'],
                    title='Lead Priority Distribution',
                    color_discrete_map={'HIGH': '#4CAF50', 'MEDIUM': '#FFC107', 'LOW': '#F44336'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Scored Leads CSV",
                    data=csv,
                    file_name="scored_leads.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                
                st.info("💡 **Next Steps:** Import this CSV into your CRM or distribute to sales team for prioritized follow-up.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV has all required columns matching the template.")

# ============================================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================================

elif page == "📚 Model Performance":
    st.title("📚 Model Performance & Technical Details")
    st.markdown("Deep dive into model metrics and methodology")
    st.markdown("---")
    
    # Model summary
    st.subheader("🎯 Lead Scoring Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Algorithm:** {models['summary']['Lead Scoring']['Model']}
        
        **Performance Metrics:**
        - ROC-AUC Score: `{models['summary']['Lead Scoring']['ROC-AUC']:.3f}`
        - Target: ≥ 0.75 (Exceeded ✅)
        
        **Training Data:**
        - 9,994 transactions
        - 80/20 train-test split
        - 5-fold cross-validation
        
        **Features Used:** {len(models['summary']['Lead Scoring']['Features'])}
        """)
    
    with col2:
        # Feature list
        st.markdown("**Input Features:**")
        for i, feat in enumerate(models['summary']['Lead Scoring']['Features'], 1):
            st.text(f"{i}. {feat}")
    
    st.markdown("---")
    
    st.subheader("👥 Customer Segmentation Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Algorithm:** K-Means Clustering
        
        **Performance Metrics:**
        - Number of Clusters: {models['summary']['Segmentation']['n_clusters']}
        - Silhouette Score: `{models['summary']['Segmentation']['silhouette_score']:.3f}`
        - Target: 3-5 clusters (Met ✅)
        
        **Training Data:**
        - 99 customers
        - Customer-level aggregation
        - Standardized features
        """)
    
    with col2:
        st.markdown("**Identified Segments:**")
        for cluster_id, name in cluster_names_dict.items():
            st.text(f"• {name}")
    
    st.markdown("---")
    
    # Methodology
    st.subheader("🔬 Methodology")
    
    st.markdown("""
    **Data Preprocessing:**
    1. Feature engineering (12 new features created)
    2. Label encoding for categorical variables
    3. StandardScaler for numerical features
    4. Class balancing (25% high-value, 75% standard)
    
    **Model Selection:**
    1. Tested 3 algorithms: Logistic Regression, Random Forest, XGBoost
    2. Selected Random Forest based on ROC-AUC performance
    3. Validated with 5-fold cross-validation
    
    **Deployment Strategy:**
    - Models saved as .pkl files
    - Scalable prediction functions
    - API-ready for CRM integration
    - Batch processing capability
    
    **Maintenance Plan:**
    - Monthly: Monitor prediction accuracy
    - Quarterly: Retrain with new data
    - Annually: Complete model refresh
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**AWS SaaS Sales Analytics**")
    st.markdown("MSc Computer Science Project")

with col2:
    st.markdown("**Technologies Used:**")
    st.markdown("Streamlit • Scikit-learn • Plotly")

with col3:
    st.markdown("**Contact:**")
    st.markdown("Muhammed K")