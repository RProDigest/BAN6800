from shiny import App, render, ui, reactive
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Load saved models (data is already scaled, no scaler needed)
try:
    kmeans_model = joblib.load("models/kmeans_model.pkl")
    print("✅ Loaded kmeans_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.joblib")
    print("✅ Loaded xgboost_model.joblib")
    explainer = shap.TreeExplainer(xgb_model)
    models_loaded = True
    print("🎉 All models loaded successfully! (No scaling required)")
except FileNotFoundError as e:
    print(f"❌ Model loading error: {e}")
    models_loaded = False
except Exception as e:
    print(f"❌ Unexpected error loading models: {e}")
    models_loaded = False

# Feature descriptions - Updated to match all model features
feature_descriptions = {
    "admin": "Time spent on administrative pages (account, checkout, etc.) in seconds",
    "prod": "Time spent browsing product pages in seconds", 
    "informational": "Time spent on informational pages in seconds",
    "intensity": "Interaction intensity score (0-1, higher = more engaged)",
    "pageval": "Average page value in Rand (economic value of pages visited)",
    "exit": "Exit rate (0-1, proportion of sessions ending on this page type)",
    "bounce": "Bounce rate (0-1, proportion of single-page sessions)",
    "special_day": "Proximity to special day (0-1, closer to holiday)",
    "weekend": "Whether session occurred on weekend (0=weekday, 1=weekend)",
    "month": "Month of the session (1-12)",
    "visitor_type": "Type of visitor (New, Returning, Other)",
    "traffic_type": "Source of traffic (Direct, Search, Social, etc.)"
}

# Super Modern Glassmorphic UI with Dataset Support
app_ui = ui.page_fillable(
    
    
    # Advanced CSS for 2025 Glassmorphic Design
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
            title="Takealot Analytics Dashboard"
),
        ui.tags.style("""
        /* 2025 Glassmorphic Design System */
        :root {
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.18);
            --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --backdrop-blur: blur(15px);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-attachment: fixed;
            min-height: 100vh;
            color: #2d3748;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
            animation: backgroundShift 15s ease-in-out infinite;
            z-index: -1;
        }
        
        @keyframes backgroundShift {
            0%, 100% { transform: translateX(0) translateY(0); }
            25% { transform: translateX(-10px) translateY(-10px); }
            50% { transform: translateX(10px) translateY(-5px); }
            75% { transform: translateX(-5px) translateY(10px); }
        }
        
        /* Glassmorphic Header */
        .glass-header {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-blur);
            -webkit-backdrop-filter: var(--backdrop-blur);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: var(--glass-shadow);
            padding: 2rem;
            margin: 1rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .dashboard-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .dashboard-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 400;
        }
        
        /* Navigation Tabs - Glassmorphic */
        .nav-tabs {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-blur);
            -webkit-backdrop-filter: var(--backdrop-blur);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 0.5rem;
            margin: 1rem;
            box-shadow: var(--glass-shadow);
        }
        
        .nav-tabs .nav-link {
            background: transparent;
            border: none;
            border-radius: 10px;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .nav-tabs .nav-link:hover {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            transform: translateY(-1px);
        }
        
        .nav-tabs .nav-link.active {
            background: var(--primary-gradient);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }
        
        /* Glassmorphic Cards */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-blur);
            -webkit-backdrop-filter: var(--backdrop-blur);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: var(--glass-shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            margin: 1rem;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(31, 38, 135, 0.2);
        }
        
        .glass-card-header {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-bottom: 1px solid var(--glass-border);
            font-weight: 600;
            font-size: 1.25rem;
            color: white;
        }
        
        .glass-card-body {
            padding: 1.5rem;
        }
        
        /* Metric Cards - Floating Effect */
        .metric-card {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-blur);
            -webkit-backdrop-filter: var(--backdrop-blur);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            margin: 1rem;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--primary-gradient);
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: -1;
        }
        
        .metric-card:hover::before {
            opacity: 0.1;
        }
        
        .metric-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
        }
        
        .metric-card.success::before {
            background: var(--success-gradient);
        }
        
        .metric-card.warning::before {
            background: var(--warning-gradient);
        }
        
        .metric-card.danger::before {
            background: var(--secondary-gradient);
        }
        
        .metric-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            margin: 1rem 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .metric-label {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            font-weight: 500;
        }
        
        .metric-subtitle {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        /* Form Controls */
        .form-label {
            color: white;
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .help-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.875rem;
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        /* Buttons - Floating Effect */
        .glass-btn {
            background: var(--primary-gradient);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .glass-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        /* Progress Bars */
        .glass-progress {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            position: relative;
        }
        
        .glass-progress-bar {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 10px;
            transition: width 0.6s ease;
        }
        
        /* Insights */
        .insight-item {
            background: rgba(255, 255, 255, 0.08);
            border-left: 4px solid;
            border-image: var(--success-gradient) 1;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: white;
            transition: all 0.3s ease;
        }
        
        .insight-item:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateX(5px);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .dashboard-title { font-size: 2rem; }
            .metric-value { font-size: 2rem; }
            .glass-card { margin: 0.5rem; }
        }
        """)
    ),
    
    # Glassmorphic Header
    ui.div(
        ui.h1("🛍️ Takealot Analytics Hub", class_="dashboard-title"),
        ui.p("Next-Gen Customer Intelligence & Predictive Analytics", class_="dashboard-subtitle"),
        class_="glass-header"
    ),
    
    # Navigation Tabs
    ui.navset_tab(
        # Tab 1: Single Customer Analysis
        ui.nav_panel("🎯 Customer Analysis",
            ui.row(
                # Input Panel
                ui.column(4,
                    ui.div(
                        ui.div("📊 Customer Behavior Inputs", class_="glass-card-header"),
                        ui.div(
                            ui.p("Enter customer session data for AI-powered analysis:", 
                                 style="color: rgba(255,255,255,0.8); margin-bottom: 1.5rem;"),
                            
                            # Administrative Duration
                            ui.div(
                                ui.tags.label("Administrative Duration (seconds)", class_="form-label"),
                                ui.input_numeric("admin", None, value=30, min=0, max=3600, step=10),
                                ui.p(feature_descriptions["admin"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Product Duration  
                            ui.div(
                                ui.tags.label("Product Browsing Duration (seconds)", class_="form-label"),
                                ui.input_numeric("prod", None, value=150, min=0, max=7200, step=10),
                                ui.p(feature_descriptions["prod"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Informational Duration
                            ui.div(
                                ui.tags.label("Informational Duration (seconds)", class_="form-label"),
                                ui.input_numeric("informational", None, value=60, min=0, max=3600, step=10),
                                ui.p(feature_descriptions["informational"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Bounce Rate
                            ui.div(
                                ui.tags.label("Bounce Rate", class_="form-label"),
                                ui.input_slider("bounce", None, min=0, max=1, value=0.01, step=0.001),
                                ui.p(feature_descriptions["bounce"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Exit Rate
                            ui.div(
                                ui.tags.label("Exit Rate", class_="form-label"),
                                ui.input_slider("exit", None, min=0, max=1, value=0.03, step=0.001),
                                ui.p(feature_descriptions["exit"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Page Values
                            ui.div(
                                ui.tags.label("Page Value (Rand)", class_="form-label"),
                                ui.input_numeric("pageval", None, value=40, min=0, max=500, step=5),
                                ui.p(feature_descriptions["pageval"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Special Day
                            ui.div(
                                ui.tags.label("Special Day Proximity", class_="form-label"),
                                ui.input_slider("special_day", None, min=0, max=1, value=0.0, step=0.1),
                                ui.p(feature_descriptions["special_day"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Weekend
                            ui.div(
                                ui.tags.label("Weekend Session", class_="form-label"),
                                ui.input_select("weekend", None, choices={"0": "Weekday", "1": "Weekend"}, selected="0"),
                                ui.p(feature_descriptions["weekend"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Month (added for completeness)
                            ui.div(
                                ui.tags.label("Month", class_="form-label"),
                                ui.input_select("month", None, 
                                              choices={str(i): f"Month {i}" for i in range(1, 13)}, 
                                              selected="6"),
                                ui.p(feature_descriptions["month"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Visitor Type (added)
                            ui.div(
                                ui.tags.label("Visitor Type", class_="form-label"),
                                ui.input_select("visitor_type", None, 
                                              choices={"new": "New Visitor", "returning": "Returning Visitor", "other": "Other"}, 
                                              selected="new"),
                                ui.p(feature_descriptions["visitor_type"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Traffic Type (added)
                            ui.div(
                                ui.tags.label("Traffic Source", class_="form-label"),
                                ui.input_select("traffic_type", None, 
                                              choices={"1": "Direct", "2": "Search Engine", "3": "Social Media", "4": "Email", "5": "Referral"}, 
                                              selected="2"),
                                ui.p(feature_descriptions["traffic_type"], class_="help-text"),
                                style="margin-bottom: 1.5rem;"
                            ),
                            
                            # Interaction Intensity
                            ui.div(
                                ui.tags.label("Interaction Intensity", class_="form-label"),
                                ui.input_slider("intensity", None, min=0, max=1, value=0.8, step=0.01),
                                ui.p(feature_descriptions["intensity"], class_="help-text"),
                                style="margin-bottom: 2rem;"
                            ),
                            
                            # Analyze Button
                            ui.input_action_button("predict_btn", "🔮 Analyze Customer", 
                                                 class_="glass-btn", 
                                                 style="width: 100%;"),
                            
                            class_="glass-card-body"
                        ),
                        class_="glass-card"
                    )
                ),
                
                # Results Panel
                ui.column(8,
                    # Metric Cards
                    ui.row(
                        ui.column(6, ui.output_ui("segmentation_card")),
                        ui.column(6, ui.output_ui("intent_card"))
                    ),
                    
                    # Analysis Charts
                    ui.row(
                        ui.column(6,
                            ui.div(
                                ui.div("🧠 AI Feature Analysis", class_="glass-card-header"),
                                ui.div(ui.output_image("shap_plot"), class_="glass-card-body"),
                                class_="glass-card"
                            )
                        ),
                        ui.column(6,
                            ui.div(
                                ui.div("📊 Behavior Visualization", class_="glass-card-header"),
                                ui.div(ui.output_ui("radar_plot"), class_="glass-card-body"),
                                class_="glass-card"
                            )
                        )
                    ),
                    
                    # Insights
                    ui.div(
                        ui.div("💡 AI-Powered Insights", class_="glass-card-header"),
                        ui.div(ui.output_ui("insights_panel"), class_="glass-card-body"),
                        class_="glass-card"
                    )
                )
            )
        ),
        
        # Tab 2: Dataset Analysis
        ui.nav_panel("📊 Dataset Analytics",
            ui.row(
                # Upload Panel
                ui.column(4,
                    ui.div(
                        ui.div("📁 Data Upload Center", class_="glass-card-header"),
                        ui.div(
                            # File Upload
                            ui.input_file("dataset_file", "Upload CSV Dataset:", 
                                        accept=[".csv"], multiple=False),
                            
                            # Quick Load
                            ui.h5("Quick Load:", style="color: white; margin: 1.5rem 0 1rem;"),
                            ui.input_action_button("load_existing", "📂 Load Demo Dataset", 
                                                 class_="glass-btn",
                                                 style="width: 100%; margin-bottom: 1rem;"),
                            ui.p("Loads: online_shoppers_Intention_cleaned.csv", 
                                 style="color: rgba(255,255,255,0.7); font-size: 0.875rem;"),
                            
                            # Analysis Settings
                            ui.h5("Analysis Settings:", style="color: white; margin: 1.5rem 0 1rem;"),
                            ui.input_numeric("sample_size", "Sample Size (0 = all):", 
                                           value=100, min=0, max=5000, step=50),
                            ui.input_action_button("analyze_batch", "🚀 Run Analysis", 
                                                 class_="glass-btn",
                                                 style="width: 100%; margin-top: 1rem;"),
                            
                            class_="glass-card-body"
                        ),
                        class_="glass-card"
                    )
                ),
                
                # Dataset Info Panel
                ui.column(8,
                    ui.div(
                        ui.div("📋 Dataset Overview", class_="glass-card-header"),
                        ui.div(ui.output_ui("dataset_info"), class_="glass-card-body"),
                        class_="glass-card"
                    ),
                    
                    ui.div(
                        ui.div("🔍 Data Preview", class_="glass-card-header"),
                        ui.div(ui.output_table("data_preview"), class_="glass-card-body"),
                        class_="glass-card"
                    ),
                    
                    ui.div(
                        ui.div("📊 Batch Analysis Results", class_="glass-card-header"),
                        ui.div(ui.output_ui("batch_results"), class_="glass-card-body"),
                        class_="glass-card"
                    )
                )
            )
        ),
        
        # Tab 3: Export Center
        ui.nav_panel("📥 Export Hub",
            ui.div(
                ui.div("📊 Analysis Summary", class_="glass-card-header"),
                ui.div(ui.output_ui("summary_stats"), class_="glass-card-body"),
                class_="glass-card"
            ),
            
            ui.div(
                ui.div("📥 Download Center", class_="glass-card-header"),
                ui.div(
                    ui.row(
                        ui.column(4,
                            ui.div(
                                ui.h5("Single Predictions", style="color: white; margin-bottom: 1rem;"),
                                ui.download_button("download_predictions", "📊 Download CSV", 
                                                 class_="glass-btn", style="width: 100%;")
                            )
                        ),
                        ui.column(4,
                            ui.div(
                                ui.h5("Batch Results", style="color: white; margin-bottom: 1rem;"),
                                ui.download_button("download_batch", "📈 Download Analysis", 
                                                 class_="glass-btn", style="width: 100%;")
                            )
                        ),
                        ui.column(4,
                            ui.div(
                                ui.h5("Full Report", style="color: white; margin-bottom: 1rem;"),
                                ui.download_button("download_report", "📋 Download Report", 
                                                 class_="glass-btn", style="width: 100%;")
                            )
                        )
                    ),
                    class_="glass-card-body"
                ),
                class_="glass-card"
            )
        )
    )
)

# Store results globally
results_log = []
batch_analysis_results = []
current_dataset = None

def server(input, output, session):
    
    # Reactive function to prepare input data - Flexible for both models
    @reactive.Calc
    def prepare_input():
        # Calculate derived features based on durations
        admin_duration = input.admin()
        prod_duration = input.prod()
        info_duration = input.informational()
        
        # Estimate page counts based on duration (rough approximation)
        admin_pages = max(1, admin_duration // 30)  # ~30 seconds per admin page
        prod_pages = max(1, prod_duration // 45)    # ~45 seconds per product page  
        info_pages = max(1, info_duration // 60)    # ~60 seconds per info page
        
        # Calculate total duration
        total_duration = admin_duration + prod_duration + info_duration
        
        # Set visitor type features
        visitor_returning = 1.0 if input.visitor_type() == "returning" else 0.0
        visitor_other = 1.0 if input.visitor_type() == "other" else 0.0
        
        # Full feature set for XGBoost (20 features) - CORRECT ORDER based on error message
        full_features = pd.DataFrame({
            # Features in the exact order expected by XGBoost model
            "Administrative": [admin_pages],
            "Administrative_Duration": [admin_duration],
            "Informational": [info_pages], 
            "Informational_Duration": [info_duration],
            "ProductRelated": [prod_pages],
            "ProductRelated_Duration": [prod_duration],
            "BounceRates": [input.bounce()],
            "ExitRates": [input.exit()],
            "PageValues": [input.pageval()],
            "SpecialDay": [input.special_day()],
            "Month": [float(input.month())],
            "OperatingSystems": [2.0],  # Common OS
            "Browser": [1.0],  # Common browser
            "Region": [3.0],  # Default region
            "TrafficType": [float(input.traffic_type())],
            "Weekend": [float(input.weekend())],
            "VisitorType_Other": [visitor_other],
            "VisitorType_Returning_Visitor": [visitor_returning],
            "Total_Duration": [total_duration],
            "Interaction_Intensity": [input.intensity()]
        })
        
        return full_features
    
    # Separate function for KMeans features (9 features)
    @reactive.Calc
    def prepare_kmeans_input():
        full_data = prepare_input()
        
        # KMeans feature subset (based on your original 9 features)
        kmeans_features = pd.DataFrame({
            "Administrative_Duration": full_data["Administrative_Duration"],
            "ProductRelated_Duration": full_data["ProductRelated_Duration"],
            "Informational_Duration": full_data["Informational_Duration"],
            "BounceRates": full_data["BounceRates"],
            "ExitRates": full_data["ExitRates"],
            "PageValues": full_data["PageValues"],
            "SpecialDay": full_data["SpecialDay"],
            "Weekend": full_data["Weekend"],
            "Interaction_Intensity": full_data["Interaction_Intensity"]
        })
        
        return kmeans_features
    
    # Load dataset reactive
    @reactive.Calc
    def get_dataset():
        global current_dataset
        
        # Check if user uploaded a file
        if input.dataset_file() is not None:
            try:
                file_info = input.dataset_file()[0]
                df = pd.read_csv(file_info["datapath"])
                current_dataset = df
                return df
            except Exception as e:
                return pd.DataFrame({"Error": [f"Failed to load file: {str(e)}"]})
        
        # Check if user clicked load existing
        elif input.load_existing() > 0:
            try:
                # Try to load the existing dataset
                df = pd.read_csv("online_shoppers_Intention_cleaned.csv")
                current_dataset = df
                return df
            except FileNotFoundError:
                return pd.DataFrame({"Error": ["online_shoppers_Intention_cleaned.csv not found in current directory"]})
            except Exception as e:
                return pd.DataFrame({"Error": [f"Failed to load existing dataset: {str(e)}"]})
        
        return None

    # Reactive predictions - Fixed to use correct features for each model
    @reactive.Calc  
    def get_predictions():
        if input.predict_btn() == 0 or not models_loaded:
            return None
            
        try:
            # Get features for both models
            xgb_data = prepare_input()  # 20 features for XGBoost
            kmeans_data = prepare_kmeans_input()  # 9 features for KMeans
            
            print(f"XGBoost data shape: {xgb_data.shape}")
            print(f"XGBoost data columns: {xgb_data.columns.tolist()}")
            print(f"KMeans data shape: {kmeans_data.shape}")
            print(f"KMeans data columns: {kmeans_data.columns.tolist()}")
            
            # Cluster prediction using KMeans features
            cluster = kmeans_model.predict(kmeans_data)[0]
            cluster_label = "High-Intent Shoppers" if cluster == 0 else "Casual Browsers"
            
            # Purchase intent prediction using XGBoost features
            purchase_prob = xgb_model.predict_proba(xgb_data)[0][1]
            purchase_pred = xgb_model.predict(xgb_data)[0]
            intent_label = "Likely to Purchase" if purchase_pred == 1 else "Unlikely to Purchase"
            
            # SHAP values using XGBoost features
            shap_values = explainer.shap_values(xgb_data)
            
            return {
                'cluster': cluster,
                'cluster_label': cluster_label,
                'purchase_prob': purchase_prob,
                'intent_label': intent_label,
                'shap_values': shap_values[0],
                'input_data': xgb_data,  # Use full feature set for display
                'kmeans_data': kmeans_data
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'error': str(e)}
    
    # Segmentation result card
    @output
    @render.ui
    def segmentation_card():
        pred = get_predictions()
        if pred is None:
            return ui.div(
                ui.span("🎯", class_="metric-icon"),
                ui.h3("Customer Segment", class_="metric-label"),
                ui.p("Click 'Analyze Customer' to see results", class_="metric-subtitle"),
                class_="metric-card"
            )
        
        if 'error' in pred:
            return ui.div(
                ui.span("❌", class_="metric-icon"),
                ui.h3("Analysis Error", class_="metric-label"),
                ui.p(f"Failed: {pred['error']}", class_="metric-subtitle"),
                class_="metric-card danger"
            )
        
        emoji = "🎯" if pred['cluster'] == 0 else "👤"
        card_class = "metric-card success" if pred['cluster'] == 0 else "metric-card warning"
        
        return ui.div(
            ui.span(emoji, class_="metric-icon"),
            ui.h2(pred['cluster_label'], class_="metric-value"),
            ui.p(f"Cluster {pred['cluster']}", class_="metric-subtitle"),
            class_=card_class
        )
    
    # Purchase intent card
    @output
    @render.ui  
    def intent_card():
        pred = get_predictions()
        if pred is None:
            return ui.div(
                ui.span("🛒", class_="metric-icon"),
                ui.h3("Purchase Intent", class_="metric-label"),
                ui.p("Click 'Analyze Customer' to see prediction", class_="metric-subtitle"),
                class_="metric-card"
            )
        
        if 'error' in pred:
            return ui.div(
                ui.span("❌", class_="metric-icon"),
                ui.h3("Analysis Error", class_="metric-label"),
                ui.p("Analysis failed", class_="metric-subtitle"),
                class_="metric-card danger"
            )
        
        emoji = "✅" if pred['purchase_prob'] > 0.5 else "❌"
        prob_percent = f"{pred['purchase_prob']*100:.1f}%"
        card_class = "metric-card success" if pred['purchase_prob'] > 0.5 else "metric-card warning"
        
        return ui.div(
            ui.span(emoji, class_="metric-icon"),
            ui.h2(pred['intent_label'], class_="metric-value"),
            ui.p(f"Probability: {prob_percent}", class_="metric-subtitle"),
            class_=card_class
        )
    
    # Dataset info display
    @output
    @render.ui
    def dataset_info():
        dataset = get_dataset()
        if dataset is None:
            return ui.div(
                ui.h5("📁 No Dataset Loaded", style="color: white; margin-bottom: 1rem;"),
                ui.p("Upload a CSV file or load the demo dataset to begin batch analysis.", 
                     style="color: rgba(255,255,255,0.8);")
            )
        
        if "Error" in dataset.columns:
            return ui.div(
                ui.h5("❌ Dataset Error", style="color: #ff6b6b; margin-bottom: 1rem;"),
                ui.p(dataset["Error"].iloc[0], style="color: rgba(255,255,255,0.8);")
            )
        
        # Dataset statistics
        rows, cols = dataset.shape
        required_cols = ["Administrative", "Administrative_Duration", "Informational", 
                        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                        "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
                        "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
                        "VisitorType_Other", "VisitorType_Returning_Visitor", "Total_Duration", 
                        "Interaction_Intensity"]
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        
        status_color = "#4ade80" if len(missing_cols) == 0 else "#fbbf24"
        status_text = "All required columns present" if len(missing_cols) == 0 else f"Missing: {missing_cols}"
        
        return ui.div(
            ui.h5("✅ Dataset Loaded Successfully", style="color: #4ade80; margin-bottom: 1rem;"),
            ui.p(f"📊 Rows: {rows:,} | Columns: {cols}", style="color: white; margin-bottom: 0.5rem;"),
            ui.p(f"🔍 Status: {status_text}", style=f"color: {status_color};")
        )
    
    # Data preview table
    @output
    @render.table
    def data_preview():
        dataset = get_dataset()
        if dataset is None or "Error" in dataset.columns:
            return pd.DataFrame({"Message": ["No valid dataset to preview"]})
        
        # Show first 10 rows of relevant columns  
        required_cols = ["Administrative", "Administrative_Duration", "Informational", 
                        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                        "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
                        "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
                        "VisitorType_Other", "VisitorType_Returning_Visitor", "Total_Duration", 
                        "Interaction_Intensity"]
        available_cols = [col for col in required_cols if col in dataset.columns]
        
        if len(available_cols) > 0:
            preview_df = dataset[available_cols].head(10).round(3)
            return preview_df
        else:
            return pd.DataFrame({"Message": ["Required columns not found in dataset"]})
    
    # Batch analysis - Fixed to use correct features for each model
    @reactive.Effect
    @reactive.event(input.analyze_batch)
    def run_batch_analysis():
        global batch_analysis_results
        dataset = get_dataset()
        
        if dataset is None or "Error" in dataset.columns or not models_loaded:
            return
        
        try:
            # Get required columns for both models - CORRECT ORDER
            xgb_cols = ["Administrative", "Administrative_Duration", "Informational", 
                       "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                       "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
                       "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
                       "VisitorType_Other", "VisitorType_Returning_Visitor", "Total_Duration", 
                       "Interaction_Intensity"]
                       
            kmeans_cols = ["Administrative_Duration", "ProductRelated_Duration", 
                          "Informational_Duration", "BounceRates", "ExitRates", 
                          "PageValues", "SpecialDay", "Weekend", "Interaction_Intensity"]
            
            # Check if we have the required columns
            missing_xgb = [col for col in xgb_cols if col not in dataset.columns]
            missing_kmeans = [col for col in kmeans_cols if col not in dataset.columns]
            
            if missing_xgb:
                print(f"Missing XGBoost columns: {missing_xgb}")
                return
            if missing_kmeans:
                print(f"Missing KMeans columns: {missing_kmeans}")
                return
            
            # Sample data if requested
            sample_size = input.sample_size()
            if sample_size > 0 and sample_size < len(dataset):
                analysis_data = dataset.sample(n=sample_size, random_state=42)
            else:
                analysis_data = dataset
            
            # Prepare features for each model
            xgb_features = analysis_data[xgb_cols]
            kmeans_features = analysis_data[kmeans_cols]
            
            # Run predictions with appropriate features
            clusters = kmeans_model.predict(kmeans_features)
            purchase_probs = xgb_model.predict_proba(xgb_features)[:, 1]
            purchase_preds = xgb_model.predict(xgb_features)
            
            # Create results dataframe
            results_df = analysis_data.copy()
            results_df["Cluster"] = clusters
            results_df["ClusterLabel"] = ["High-Intent" if c == 0 else "Casual Browser" for c in clusters]
            results_df["PurchaseProbability"] = purchase_probs
            results_df["PurchaseIntent"] = ["Likely" if p == 1 else "Unlikely" for p in purchase_preds]
            results_df["Timestamp"] = datetime.now().isoformat()
            
            batch_analysis_results = [results_df]
            
        except Exception as e:
            print(f"Batch analysis error: {e}")
    
    # Batch results display
    @output
    @render.ui  
    def batch_results():
        if len(batch_analysis_results) == 0:
            return ui.p("No batch analysis results yet. Click 'Run Analysis' to start.", 
                       style="color: rgba(255,255,255,0.8);")
        
        df = batch_analysis_results[0]
        total_rows = len(df)
        
        # Summary statistics
        high_intent_count = sum(df["Cluster"] == 0)
        likely_purchase_count = sum(df["PurchaseIntent"] == "Likely")
        avg_purchase_prob = df["PurchaseProbability"].mean()
        
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.div(
                        ui.span("📊", class_="metric-icon"),
                        ui.h2(f"{total_rows:,}", class_="metric-value"),
                        ui.p("Total Analyzed", class_="metric-label"),
                        class_="metric-card"
                    )
                ),
                ui.column(3,
                    ui.div(
                        ui.span("🎯", class_="metric-icon"),
                        ui.h2(f"{high_intent_count:,}", class_="metric-value"),
                        ui.p(f"{high_intent_count/total_rows*100:.1f}% High-Intent", class_="metric-label"),
                        class_="metric-card success"
                    )
                ),
                ui.column(3,
                    ui.div(
                        ui.span("🛒", class_="metric-icon"),
                        ui.h2(f"{likely_purchase_count:,}", class_="metric-value"),
                        ui.p(f"{likely_purchase_count/total_rows*100:.1f}% Likely Purchase", class_="metric-label"),
                        class_="metric-card warning"
                    )
                ),
                ui.column(3,
                    ui.div(
                        ui.span("📈", class_="metric-icon"),
                        ui.h2(f"{avg_purchase_prob:.1%}", class_="metric-value"),
                        ui.p("Avg Purchase Prob", class_="metric-label"),
                        class_="metric-card"
                    )
                )
            ),
            ui.div(
                ui.h6("Recent Results Preview:", style="color: white; margin: 1.5rem 0 1rem;"),
                ui.output_table("batch_preview_table")
            )
        )
    
    # Batch preview table
    @output
    @render.table
    def batch_preview_table():
        if len(batch_analysis_results) > 0:
            df = batch_analysis_results[0]
            return df[["ClusterLabel", "PurchaseIntent", "PurchaseProbability", 
                      "Administrative_Duration", "ProductRelated_Duration"]].head()
        else:
            return pd.DataFrame({"Message": ["No batch results available"]})
    
    # SHAP plot - Fixed scope and import issues
    @output
    @render.image
    def shap_plot():
        pred = get_predictions()
        if pred is None or 'error' in pred:
            # Create a simple placeholder image when no data is available
            try:
                import tempfile
                
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                # Create placeholder text with simpler styling
                ax.text(0.5, 0.5, 'Click "Analyze Customer" to see\nfeature importance analysis', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=16, color='white', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='gray', 
                                edgecolor='white', alpha=0.3))
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Remove all spines and ticks
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                
                plt.tight_layout()
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, format="png", dpi=150, bbox_inches='tight', 
                               facecolor='none', edgecolor='none', transparent=True)
                    plt.close(fig)
                    temp_filename = tmp.name
                
                return {"src": temp_filename, "alt": "SHAP placeholder", "width": "100%"}
                
            except Exception as e:
                print(f"Error creating placeholder: {e}")
                # Return a completely empty response to avoid file errors
                return None
        
        try:
            # Create figure with proper settings
            plt.ioff()  # Turn off interactive mode
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            feature_names = prepare_input().columns.tolist()
            shap_vals = pred['shap_values']
            
            # Ensure we have the right number of features
            if len(shap_vals) != len(feature_names):
                print(f"SHAP values length: {len(shap_vals)}, Feature names length: {len(feature_names)}")
                plt.close(fig)
                return None
            
            colors = ['#ff6b6b' if x < 0 else '#4ade80' for x in shap_vals]
            
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name.replace('_', ' ') for name in feature_names], color='white', fontsize=10)
            ax.set_xlabel('SHAP Value (Impact on Purchase Intent)', color='white', fontsize=12)
            ax.set_title('AI Feature Importance Analysis', fontsize=14, fontweight='bold', color='white')
            ax.axvline(x=0, color='white', linestyle='-', alpha=0.3)
            
            # Style the plot for glassmorphic theme
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='white')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, shap_vals)):
                ax.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                       va='center', ha='left' if val >= 0 else 'right', 
                       fontweight='bold', color='white', fontsize=9)
            
            plt.tight_layout()
            
            # Use a different approach for Shiny - save to a temporary file
            import tempfile
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, format="png", dpi=150, bbox_inches='tight', 
                           facecolor='none', edgecolor='none', transparent=True)
                plt.close(fig)
                temp_filename = tmp.name
            
            # Return the path to the temporary file
            return {"src": temp_filename, "alt": "SHAP Feature Importance", "width": "100%"}
            
        except Exception as e:
            print(f"SHAP plot error: {e}")
            # Clean up any open figures - plt is in scope here since it's imported at module level
            try:
                plt.close('all')
            except:
                pass
            return None
    
    # Radar chart for input visualization
    @output
    @render.ui
    def radar_plot():
        pred = get_predictions()
        if pred is None or 'error' in pred:
            return ui.p("No data to visualize", style="color: rgba(255,255,255,0.8);")
        
        try:
            input_data = pred['input_data'].iloc[0]
            
            # Define max values for normalization - updated for key features
            max_vals = {
                'Administrative_Duration': 300,
                'ProductRelated_Duration': 1000, 
                'Informational_Duration': 500,
                'Administrative': 10,
                'ProductRelated': 30,
                'Informational': 15,
                'BounceRates': 0.2,
                'ExitRates': 0.2,
                'PageValues': 100,
                'SpecialDay': 1.0,
                'Weekend': 1.0,
                'Month': 12.0,
                'OperatingSystems': 8.0,
                'Browser': 13.0,
                'Region': 9.0,
                'TrafficType': 20.0,
                'VisitorType_Other': 1.0,
                'VisitorType_Returning_Visitor': 1.0,
                'Total_Duration': 2000,
                'Interaction_Intensity': 1.0
            }
            
            normalized_vals = []
            labels = []
            for col in input_data.index:
                normalized = min(input_data[col] / max_vals.get(col, 1), 1.0)
                normalized_vals.append(normalized)
                labels.append(col.replace('_', ' '))
            
            # Create glassmorphic progress bars
            bars_html = ""
            for label, val in zip(labels, normalized_vals):
                bars_html += f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: white; font-weight: 500;">{label}</span>
                        <span style="color: rgba(255,255,255,0.8);">{val:.2f}</span>
                    </div>
                    <div class="glass-progress">
                        <div class="glass-progress-bar" style="width: {val*100}%;"></div>
                    </div>
                </div>
                """
            
            return ui.HTML(f"""
            <div style="padding: 1rem;">
                <h6 style="color: white; text-align: center; margin-bottom: 1.5rem;">Behavior Metrics (Normalized)</h6>
                {bars_html}
            </div>
            """)
            
        except Exception as e:
            return ui.p(f"Visualization error: {str(e)}", style="color: #ff6b6b;")
    
    # Customer insights panel
    @output
    @render.ui
    def insights_panel():
        pred = get_predictions()
        if pred is None:
            return ui.p("Run analysis to see AI-powered insights", style="color: rgba(255,255,255,0.8);")
        
        if 'error' in pred:
            return ui.p("Unable to generate insights due to analysis error", style="color: #ff6b6b;")
        
        # Generate insights based on predictions
        insights = []
        input_data = pred['input_data'].iloc[0]  # Use full feature set for insights
        
        # Behavioral insights - updated for all features
        if input_data['ProductRelated_Duration'] > 300:
            insights.append("🔍 High product engagement - customer is actively researching")
        elif input_data['ProductRelated_Duration'] < 60:
            insights.append("⚡ Quick browsing pattern - may need targeted recommendations")
        
        if input_data['PageValues'] > 50:
            insights.append("💰 High-value page interactions - strong commercial intent")
        elif input_data['PageValues'] < 10:
            insights.append("👀 Low commercial engagement - consider promotional offers")
        
        if input_data['ExitRates'] > 0.1:
            insights.append("🚪 High exit rate - potential UX issues or price sensitivity")
        elif input_data['ExitRates'] < 0.02:
            insights.append("✨ Low exit rate - engaged customer journey")
            
        if input_data['BounceRates'] > 0.1:
            insights.append("⚠️ High bounce rate - improve landing page experience")
        elif input_data['BounceRates'] < 0.01:
            insights.append("🎯 Low bounce rate - excellent page engagement")
            
        if input_data['Weekend'] == 1:
            insights.append("📅 Weekend session - leisure browsing pattern detected")
        else:
            insights.append("💼 Weekday session - work-time browsing behavior")
            
        if input_data['SpecialDay'] > 0.5:
            insights.append("🎁 Shopping near special day - holiday influence detected")
        
        # Recommendations based on segment
        if pred['cluster'] == 0:  # High-intent
            insights.append("🎯 Recommendation: Send targeted product offers and limited-time discounts")
            insights.append("📧 Consider email retargeting with abandoned cart reminders")
        else:  # Low-intent
            insights.append("🌟 Recommendation: Focus on brand awareness and educational content")
            insights.append("🎁 Consider welcome offers or loyalty program enrollment")
        
        if not insights:
            insights = ["📊 Customer shows typical browsing patterns"]
        
        insights_html = ""
        for insight in insights:
            insights_html += f'<div class="insight-item">{insight}</div>'
        
        return ui.HTML(insights_html)
    
    # Summary statistics for export tab
    @output
    @render.ui
    def summary_stats():
        single_count = len(results_log)
        batch_count = len(batch_analysis_results[0]) if len(batch_analysis_results) > 0 else 0
        
        return ui.row(
            ui.column(4,
                ui.div(
                    ui.span("👤", class_="metric-icon"),
                    ui.h2(f"{single_count}", class_="metric-value"),
                    ui.p("Single Predictions", class_="metric-label"),
                    class_="metric-card"
                )
            ),
            ui.column(4,
                ui.div(
                    ui.span("📊", class_="metric-icon"),
                    ui.h2(f"{batch_count:,}", class_="metric-value"),
                    ui.p("Batch Predictions", class_="metric-label"),
                    class_="metric-card success"
                )
            ),
            ui.column(4,
                ui.div(
                    ui.span("🎯", class_="metric-icon"),
                    ui.h2(f"{single_count + batch_count:,}", class_="metric-value"),
                    ui.p("Total Analysis", class_="metric-label"),
                    class_="metric-card warning"
                )
            )
        )
    
    # Log predictions for download
    @reactive.Effect
    def log_prediction():
        pred = get_predictions()
        if pred and 'error' not in pred:
            record = prepare_kmeans_input().copy()  # Use KMeans features for consistency
            record["Cluster"] = [pred['cluster_label']]
            record["ClusterID"] = [pred['cluster']]
            record["PurchaseIntent"] = [pred['intent_label']]
            record["PurchaseProbability"] = [f"{pred['purchase_prob']:.3f}"]
            record["Timestamp"] = [datetime.now().isoformat()]
            results_log.append(record)
    
    # FIXED DOWNLOAD FUNCTIONS - Using proper file handling
    @render.download(filename=lambda: f"takealot_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    def download_predictions():
        def write_csv():
            if results_log:
                df = pd.concat(results_log, ignore_index=True)
                yield df.to_csv(index=False)
            else:
                # Return empty CSV with headers when no data
                empty_df = pd.DataFrame(columns=["Administrative_Duration", "ProductRelated_Duration", 
                                               "Informational_Duration", "BounceRates", "ExitRates", 
                                               "PageValues", "SpecialDay", "Weekend", "Interaction_Intensity",
                                               "Cluster", "ClusterID", "PurchaseIntent", "PurchaseProbability", "Timestamp"])
                yield empty_df.to_csv(index=False)
        
        return write_csv()
    
    @render.download(filename=lambda: f"takealot_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    def download_batch():
        def write_csv():
            if len(batch_analysis_results) > 0:
                yield batch_analysis_results[0].to_csv(index=False)
            else:
                # Return empty CSV with basic headers when no data
                empty_df = pd.DataFrame(columns=["ClusterLabel", "PurchaseIntent", "PurchaseProbability", 
                                               "Administrative_Duration", "ProductRelated_Duration", "Timestamp"])
                yield empty_df.to_csv(index=False)
        
        return write_csv()
    
    @render.download(filename=lambda: f"takealot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    def download_report():
        def write_report():
            pred = get_predictions()
            if pred is None or 'error' in pred:
                # Return a proper text file content when no analysis is available
                report_content = f"""TAKEALOT CUSTOMER ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== STATUS ===
No analysis results available.
Please run a customer analysis first by:
1. Entering customer behavior data in the input fields
2. Clicking the 'Analyze Customer' button
3. Returning to this tab to download the report

=== INSTRUCTIONS ===
This report will contain:
- Customer behavior profile
- Segmentation results (High-Intent vs Casual Browser)
- Purchase intent prediction
- Feature importance analysis (SHAP values)
- AI-powered insights and recommendations

Generated by Takealot Customer Analytics Dashboard
"""
                yield report_content
            else:
                input_data = pred['input_data'].iloc[0]
                
                # Build the report content properly
                report_lines = [
                    "TAKEALOT CUSTOMER ANALYTICS REPORT",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "=== CUSTOMER PROFILE ===",
                    f"Administrative Duration: {input_data['Administrative_Duration']:.1f} seconds",
                    f"Product Browsing Duration: {input_data['ProductRelated_Duration']:.1f} seconds",
                    f"Informational Duration: {input_data['Informational_Duration']:.1f} seconds",
                    f"Bounce Rate: {input_data['BounceRates']:.3f}",
                    f"Exit Rate: {input_data['ExitRates']:.3f}",
                    f"Page Value: R{input_data['PageValues']:.2f}",
                    f"Special Day: {input_data['SpecialDay']:.1f}",
                    f"Weekend: {'Yes' if input_data['Weekend'] == 1 else 'No'}",
                    f"Interaction Intensity: {input_data['Interaction_Intensity']:.2f}",
                    "",
                    "=== ANALYSIS RESULTS ===",
                    f"Customer Segment: {pred['cluster_label']} (Cluster {pred['cluster']})",
                    f"Purchase Intent: {pred['intent_label']}",
                    f"Purchase Probability: {pred['purchase_prob']:.1%}",
                    "",
                    "=== FEATURE IMPORTANCE (SHAP VALUES) ==="
                ]
                
                feature_names = prepare_input().columns.tolist()
                for name, shap_val in zip(feature_names, pred['shap_values']):
                    impact = "Positive" if shap_val > 0 else "Negative"
                    report_lines.append(f"{name}: {shap_val:.4f} ({impact} impact)")
                
                report_lines.extend([
                    "",
                    "=== SUMMARY ===",
                    f"This customer belongs to the {pred['cluster_label']} segment and is {pred['intent_label'].lower()}.",
                    f"The model confidence is {pred['purchase_prob']:.1%}.",
                    "",
                    "Generated by Takealot Customer Analytics Dashboard"
                ])
                
                # Join all lines with newlines to create the final report
                yield "\n".join(report_lines)
        
        return write_report()

# Create the app
app = App(app_ui, server)

# Run the app when script is executed directly
if __name__ == "__main__":
    app.run()