import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom ML models
from advanced_ml_models import (
    HybridRecommenderEngine,
    TextSimilarityEngine,
    CompetencyClusteringModel,
    CollaborativeFilteringRecommender,
    prepare_training_catalog
)

# Page configuration
st.set_page_config(
    page_title="AI Training Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #28a745;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .priority-critical {
        border-left-color: #dc3545 !important;
    }
    .priority-high {
        border-left-color: #fd7e14 !important;
    }
    .priority-medium {
        border-left-color: #ffc107 !important;
    }
    .priority-low {
        border-left-color: #28a745 !important;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class IntegratedTrainingRecommender:
    """
    Integrated Training Recommender with ML Models and Training Catalog
    """
    
    def __init__(self, employee_data_path=None, training_catalog_path=None):
        self.employee_df = None
        self.training_catalog = None
        self.hybrid_engine = None
        self.competency_columns = []
        self.fitted = False
        
        if employee_data_path:
            self.load_employee_data(employee_data_path)
        if training_catalog_path:
            self.load_training_catalog(training_catalog_path)
    
    def load_employee_data(self, file_path):
        """Load employee competency data"""
        self.employee_df = pd.read_excel(file_path)
        self.competency_columns = [col for col in self.employee_df.columns 
                                   if col.startswith(('core_', 'managerial_', 'leadership_'))]
        return self
    
    def load_training_catalog(self, file_path):
        """Load training catalog"""
        self.training_catalog = pd.read_excel(file_path)
        self.training_catalog = prepare_training_catalog(self.training_catalog)
        return self
    
    def fit_models(self):
        """Fit all ML models"""
        if self.employee_df is None or self.training_catalog is None:
            raise ValueError("Load both employee data and training catalog first")
        
        with st.spinner("🤖 Training ML models..."):
            self.hybrid_engine = HybridRecommenderEngine()
            self.hybrid_engine.fit(self.employee_df, self.training_catalog)
            self.fitted = True
        
        st.success("✅ ML models trained successfully!")
        return self
    
    def get_recommendations(self, employee_id, top_n=10, filters=None):
        """Generate recommendations using hybrid ML approach"""
        if not self.fitted:
            raise ValueError("Models not fitted. Call fit_models() first.")
        
        recommendations = self.hybrid_engine.generate_recommendations(
            employee_id=employee_id,
            employee_data=self.employee_df,
            training_catalog=self.training_catalog,
            top_n=top_n,
            filters=filters
        )
        
        return recommendations
    
    def get_employee_summary(self, employee_id):
        """Get comprehensive employee summary"""
        employee = self.employee_df[self.employee_df['employee_id'] == employee_id].iloc[0]
        
        # Calculate competency categories
        core_comps = [col for col in self.competency_columns if col.startswith('core_')]
        managerial_comps = [col for col in self.competency_columns if col.startswith('managerial_')]
        leadership_comps = [col for col in self.competency_columns if col.startswith('leadership_')]
        
        summary = {
            'employee_id': employee_id,
            'name': employee['name'],
            'position': employee['current_position'],
            'division': employee['division'],
            'experience_level': employee['experience_level'],
            'core_avg': employee[core_comps].mean(),
            'managerial_avg': employee[managerial_comps].mean(),
            'leadership_avg': employee[leadership_comps].mean(),
            'overall_avg': employee[self.competency_columns].mean(),
            'weakest_competencies': self._get_weakest_comps(employee, n=5),
            'strongest_competencies': self._get_strongest_comps(employee, n=3),
            'cluster': employee.get('cluster', None)
        }
        
        return summary
    
    def _get_weakest_comps(self, employee, n=5):
        """Get weakest competencies"""
        comp_scores = {col: employee[col] for col in self.competency_columns}
        sorted_comps = sorted(comp_scores.items(), key=lambda x: x[1])
        return sorted_comps[:n]
    
    def _get_strongest_comps(self, employee, n=3):
        """Get strongest competencies"""
        comp_scores = {col: employee[col] for col in self.competency_columns}
        sorted_comps = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_comps[:n]
    
    def get_training_details(self, training_id):
        """Get detailed training information"""
        training = self.training_catalog[
            self.training_catalog['training_id'] == training_id
        ].iloc[0]
        
        return training.to_dict()
    
    def analyze_batch(self, filter_type='division', filter_value=None):
        """Batch analysis for groups"""
        if filter_type == 'division' and filter_value:
            filtered = self.employee_df[self.employee_df['division'] == filter_value]
        elif filter_type == 'level' and filter_value:
            filtered = self.employee_df[self.employee_df['experience_level'] == filter_value]
        else:
            filtered = self.employee_df
        
        # Calculate aggregate gaps
        gap_summary = {}
        for comp in self.competency_columns:
            avg_score = filtered[comp].mean()
            gap = max(0, 4.0 - avg_score)
            gap_summary[comp] = {
                'avg_score': avg_score,
                'gap': gap,
                'std': filtered[comp].std()
            }
        
        return gap_summary, filtered


# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'models_fitted' not in st.session_state:
    st.session_state.models_fitted = False


def main():
    """Main application"""
    st.markdown("<h1 class='main-header'>🎓 AI-Powered Training Recommender System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Intelligent Training Recommendations using Advanced Machine Learning</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # File uploads
        st.subheader("📁 Data Upload")
        
        employee_file = st.file_uploader(
            "Upload Employee Data (Excel)",
            type=['xlsx'],
            help="File containing employee competency assessments"
        )
        
        training_file = st.file_uploader(
            "Upload Training Catalog (Excel)",
            type=['xlsx'],
            help="File containing available training programs"
        )
        
        # Initialize recommender
        if employee_file and training_file:
            if st.session_state.recommender is None:
                try:
                    st.session_state.recommender = IntegratedTrainingRecommender(
                        employee_file, training_file
                    )
                    st.success("✅ Data loaded successfully!")
                    
                    # Display info
                    st.info(f"👥 Employees: {len(st.session_state.recommender.employee_df)}")
                    st.info(f"📚 Trainings: {len(st.session_state.recommender.training_catalog)}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
            
            # Fit models button
            if not st.session_state.models_fitted:
                if st.button("🚀 Train ML Models", type="primary"):
                    st.session_state.recommender.fit_models()
                    st.session_state.models_fitted = True
                    st.rerun()
        
        st.markdown("---")
        
        # Model status
        if st.session_state.models_fitted:
            st.success("🤖 ML Models: Ready")
        else:
            st.warning("⏳ ML Models: Not trained")
        
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("- TF-IDF + Cosine Similarity")
        st.markdown("- K-Means Clustering")
        st.markdown("- Matrix Factorization (NMF)")
        st.markdown("- Hybrid Recommendation")
    
    # Main content
    if st.session_state.recommender is None:
        show_welcome_screen()
        return
    
    if not st.session_state.models_fitted:
        st.warning("⚠️ Please train ML models first using the sidebar button")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Smart Recommendations",
        "📊 Analytics Dashboard", 
        "👤 Employee Deep Dive",
        "📈 Batch Analysis",
        "🔍 Training Catalog"
    ])
    
    with tab1:
        show_smart_recommendations()
    
    with tab2:
        show_analytics_dashboard()
    
    with tab3:
        show_employee_deep_dive()
    
    with tab4:
        show_batch_analysis()
    
    with tab5:
        show_training_catalog()


def show_welcome_screen():
    """Welcome screen with instructions"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👋 Welcome to AI Training Recommender!")
        st.markdown("---")
        
        st.markdown("""
        #### 🚀 Getting Started
        
        1. **Upload Employee Data** - Excel file with competency assessments
        2. **Upload Training Catalog** - Excel file with training programs
        3. **Train ML Models** - Click the button in sidebar
        4. **Get Recommendations** - Start exploring!
        
        #### 🧠 AI Technologies
        
        This system uses cutting-edge machine learning:
        
        - **TF-IDF Vectorization** - Text analysis of training objectives
        - **Cosine Similarity** - Semantic matching
        - **K-Means Clustering** - Employee segmentation
        - **NMF Matrix Factorization** - Collaborative filtering
        - **Hybrid Algorithm** - Multi-model ensemble
        
        #### 📊 Key Features
        
        ✨ Personalized recommendations based on competency gaps  
        ✨ Content-based filtering using training objectives  
        ✨ Collaborative filtering from similar employees  
        ✨ Cost-benefit analysis and ROI estimation  
        ✨ PESTLE-aware strategic alignment  
        ✨ Batch analysis for team planning  
        """)


def show_smart_recommendations():
    """Smart recommendations tab with ML-powered suggestions"""
    st.header("🎯 AI-Powered Training Recommendations")
    
    recommender = st.session_state.recommender
    
    # Employee selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        employee_options = recommender.employee_df['employee_id'].tolist()
        selected_employee = st.selectbox(
            "Select Employee",
            options=employee_options,
            format_func=lambda x: f"{x} - {recommender.employee_df[recommender.employee_df['employee_id']==x]['name'].iloc[0]} ({recommender.employee_df[recommender.employee_df['employee_id']==x]['current_position'].iloc[0]})"
        )
    
    with col2:
        top_n = st.slider("# Recommendations", 5, 20, 10)
    
    # Advanced filters
    with st.expander("🔍 Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_cost = st.number_input(
                "Max Cost (IDR)",
                min_value=0,
                max_value=50000000,
                value=20000000,
                step=1000000
            )
        
        with col2:
            max_duration = st.number_input(
                "Max Duration (days)",
                min_value=1,
                max_value=30,
                value=10
            )
        
        with col3:
            job_families = ['All'] + recommender.training_catalog['job_family'].unique().tolist()
            job_family_filter = st.selectbox("Job Family", job_families)
    
    # Generate recommendations
    if st.button("🔮 Generate AI Recommendations", type="primary"):
        with st.spinner("🤖 AI is analyzing competencies and matching trainings..."):
            filters = {
                'max_cost': max_cost,
                'max_duration': max_duration
            }
            if job_family_filter != 'All':
                filters['job_family'] = job_family_filter
            
            recommendations = recommender.get_recommendations(
                selected_employee,
                top_n=top_n,
                filters=filters
            )
            
            # Employee summary
            summary = recommender.get_employee_summary(selected_employee)
            
            st.markdown("---")
            
            # Display employee info
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Employee", summary['name'])
            with col2:
                st.metric("Level", summary['experience_level'])
            with col3:
                st.metric("Core Avg", f"{summary['core_avg']:.2f}")
            with col4:
                st.metric("Managerial Avg", f"{summary['managerial_avg']:.2f}")
            with col5:
                st.metric("Leadership Avg", f"{summary['leadership_avg']:.2f}")
            
            st.markdown("---")
            
            # Display recommendations
            if recommendations:
                st.subheader(f"📚 Top {len(recommendations)} AI-Recommended Trainings")
                
                for idx, rec in enumerate(recommendations, 1):
                    priority_class = f"priority-{rec['priority'].lower()}"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class='recommendation-card {priority_class}'>
                            <h3>#{idx} {rec['training_name']}</h3>
                            <p style='color: #666; margin: 0.5rem 0;'><strong>ID:</strong> {rec['training_id']} | <strong>School:</strong> {rec['school']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**📖 Learning Objectives:**")
                            st.write(rec['learning_objectives'])
                            
                            st.markdown("**🎯 Target:**")
                            st.write(f"Division: {rec['target_division']} | Level: {rec['target_level']}")
                            
                            st.markdown("**💡 Job Family:** " + rec['job_family'])
                        
                        with col2:
                            # Priority badge
                            priority_colors = {
                                'Critical': '#dc3545',
                                'High': '#fd7e14',
                                'Medium': '#ffc107',
                                'Low': '#28a745'
                            }
                            st.markdown(f"""
                            <div style='background: {priority_colors[rec['priority']]}; 
                                        color: white; 
                                        padding: 0.5rem; 
                                        border-radius: 8px; 
                                        text-align: center;
                                        font-weight: bold;
                                        margin-bottom: 1rem;'>
                                {rec['priority']} Priority
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics
                            st.metric("Final Score", f"{rec['final_score']:.2f}/5.0")
                            st.metric("Duration", f"{rec['duration_days']} days")
                            st.metric("Cost", f"Rp {rec['cost']:,.0f}")
                            st.metric("ROI Estimate", f"{rec['roi_estimate']:.1f}%")
                        
                        # Score breakdown
                        with st.expander("📊 Detailed Score Breakdown"):
                            scores = rec['score_breakdown']
                            
                            score_df = pd.DataFrame([
                                {"Component": "Content-Based (TF-IDF)", "Score": scores['content_based']},
                                {"Component": "Competency Gap", "Score": scores['competency_gap']},
                                {"Component": "Collaborative Filtering", "Score": scores['collaborative']},
                                {"Component": "PESTLE Relevance", "Score": scores['pestle']},
                                {"Component": "Cluster-Based", "Score": scores['cluster_based']},
                                {"Component": "Cost Efficiency", "Score": scores['cost_efficiency']}
                            ])
                            
                            fig = px.bar(score_df, x='Score', y='Component', orientation='h',
                                       title="Score Components",
                                       color='Score',
                                       color_continuous_scale='Viridis')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
            else:
                st.warning("No recommendations found with current filters. Try adjusting the criteria.")


def show_analytics_dashboard():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    recommender = st.session_state.recommender
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(recommender.employee_df))
    with col2:
        st.metric("Total Trainings", len(recommender.training_catalog))
    with col3:
        avg_comp = recommender.employee_df[recommender.competency_columns].mean().mean()
        st.metric("Avg Competency", f"{avg_comp:.2f}")
    with col4:
        total_budget = recommender.training_catalog['cost'].sum()
        st.metric("Total Training Budget", f"Rp {total_budget/1e9:.1f}B")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Training by job family
        job_family_dist = recommender.training_catalog['job_family'].value_counts()
        fig = px.pie(values=job_family_dist.values, names=job_family_dist.index,
                    title="Training Distribution by Job Family",
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost distribution
        fig = px.box(recommender.training_catalog, y='cost',
                    title="Training Cost Distribution",
                    labels={'cost': 'Cost (IDR)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Duration vs Cost scatter
    fig = px.scatter(recommender.training_catalog, 
                    x='duration_days', 
                    y='cost',
                    size='cost',
                    color='job_family',
                    hover_data=['training_name'],
                    title="Training Duration vs Cost Analysis",
                    labels={'duration_days': 'Duration (days)', 'cost': 'Cost (IDR)'})
    st.plotly_chart(fig, use_container_width=True)


def show_employee_deep_dive():
    """Employee deep dive analysis"""
    st.header("👤 Employee Deep Dive Analysis")
    
    recommender = st.session_state.recommender
    
    # Employee selection
    employee_options = recommender.employee_df['employee_id'].tolist()
    selected_employee = st.selectbox(
        "Select Employee for Deep Analysis",
        options=employee_options,
        format_func=lambda x: f"{x} - {recommender.employee_df[recommender.employee_df['employee_id']==x]['name'].iloc[0]}"
    )
    
    summary = recommender.get_employee_summary(selected_employee)
    employee = recommender.employee_df[recommender.employee_df['employee_id'] == selected_employee].iloc[0]
    
    # Profile display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Profile")
        st.markdown(f"**Name:** {summary['name']}")
        st.markdown(f"**ID:** {summary['employee_id']}")
        st.markdown(f"**Position:** {summary['position']}")
        st.markdown(f"**Division:** {summary['division']}")
        st.markdown(f"**Level:** {summary['experience_level']}")
        if summary['cluster'] is not None:
            st.markdown(f"**Cluster:** {summary['cluster']}")
    
    with col2:
        # Competency category overview
        categories = ['Core', 'Managerial', 'Leadership']
        scores = [summary['core_avg'], summary['managerial_avg'], summary['leadership_avg']]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=scores, 
                  text=[f"{s:.2f}" for s in scores],
                  textposition='auto',
                  marker_color=['#3498db', '#e74c3c', '#2ecc71'])
        ])
        fig.update_layout(
            title="Competency Category Averages",
            yaxis_title="Score",
            yaxis_range=[0, 5],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed competency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Development Areas")
        for comp, score in summary['weakest_competencies']:
            comp_name = comp.replace('_', ' ').title()
            progress = score / 5.0
            st.markdown(f"**{comp_name}**")
            st.progress(progress)
            st.caption(f"Score: {score:.2f}/5.0 | Gap: {max(0, 4.0-score):.2f}")
    
    with col2:
        st.markdown("### 🟢 Strengths")
        for comp, score in summary['strongest_competencies']:
            comp_name = comp.replace('_', ' ').title()
            progress = score / 5.0
            st.markdown(f"**{comp_name}**")
            st.progress(progress)
            st.caption(f"Score: {score:.2f}/5.0")
    
    # Spider chart
    st.markdown("---")
    st.markdown("### 🕸️ Complete Competency Profile")
    
    comp_data = [employee[col] for col in recommender.competency_columns]
    comp_names = [col.replace('_', ' ').title() for col in recommender.competency_columns]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=comp_data,
        theta=comp_names,
        fill='toself',
        name=summary['name']
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def show_batch_analysis():
    """Batch analysis tab"""
    st.header("📈 Batch Training Needs Analysis")
    
    recommender = st.session_state.recommender
    
    # Analysis type selection
    analysis_type = st.radio(
        "Select Analysis Scope",
        ["By Division", "By Experience Level", "All Employees"],
        horizontal=True
    )
    
    if analysis_type == "By Division":
        divisions = recommender.employee_df['division'].unique().tolist()
        selected = st.selectbox("Select Division", divisions)
        filter_type, filter_value = 'division', selected
    elif analysis_type == "By Experience Level":
        levels = recommender.employee_df['experience_level'].unique().tolist()
        selected = st.selectbox("Select Experience Level", levels)
        filter_type, filter_value = 'level', selected
    else:
        filter_type, filter_value = None, None
    
    if st.button("🔍 Analyze Group Needs", type="primary"):
        with st.spinner("Analyzing group training needs..."):
            gap_summary, filtered_df = recommender.analyze_batch(filter_type, filter_value)
            
            st.metric("Employees in Selection", len(filtered_df))
            
            st.markdown("---")
            st.subheader("📊 Aggregated Competency Gaps")
            
            # Convert to DataFrame for visualization
            gap_df = pd.DataFrame([
                {
                    'Competency': k.replace('_', ' ').title(),
                    'Avg Score': v['avg_score'],
                    'Gap': v['gap'],
                    'Std Dev': v['std']
                }
                for k, v in gap_summary.items()
            ]).sort_values('Gap', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(gap_df.head(10), use_container_width=True)
            
            with col2:
                fig = px.bar(gap_df.head(10), 
                           x='Gap', 
                           y='Competency',
                           orientation='h',
                           title="Top 10 Competency Gaps",
                           color='Gap',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)


def show_training_catalog():
    """Training catalog browser"""
    st.header("🔍 Training Catalog Explorer")
    
    recommender = st.session_state.recommender
    
    # Search and filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔎 Search trainings", "")
    
    with col2:
        job_families = ['All'] + recommender.training_catalog['job_family'].unique().tolist()
        job_filter = st.selectbox("Filter by Job Family", job_families)
    
    with col3:
        sort_by = st.selectbox("Sort by", ['Cost', 'Duration', 'Training Name'])
    
    # Filter catalog
    filtered_catalog = recommender.training_catalog.copy()
    
    if search_term:
        filtered_catalog = filtered_catalog[
            filtered_catalog['training_name'].str.contains(search_term, case=False, na=False) |
            filtered_catalog['learning_objectives'].str.contains(search_term, case=False, na=False)
        ]
    
    if job_filter != 'All':
        filtered_catalog = filtered_catalog[filtered_catalog['job_family'] == job_filter]
    
    # Sort
    sort_col_map = {'Cost': 'cost', 'Duration': 'duration_days', 'Training Name': 'training_name'}
    filtered_catalog = filtered_catalog.sort_values(sort_col_map[sort_by])
    
    st.info(f"Showing {len(filtered_catalog)} trainings")
    
    # Display catalog
    for _, training in filtered_catalog.iterrows():
        with st.expander(f"{training['training_name']} ({training['training_id']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**School:** {training['school']}")
                st.markdown(f"**Learning Objectives:**")
                st.write(training['learning_objectives'])
                st.markdown(f"**Target Division:** {training['target_division']}")
                st.markdown(f"**Target Level:** {training['target_level']}")
            
            with col2:
                st.metric("Duration", f"{training['duration_days']} days")
                st.metric("Cost", f"Rp {training['cost']:,.0f}")
                st.markdown(f"**Job Family:** {training['job_family']}")


if __name__ == "__main__":
    main()