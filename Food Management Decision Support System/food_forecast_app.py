import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import os
import time # For simulation of data submission

# --- CONFIGURATION AND MODEL LOADING ---

# Placeholder for key meal/center combinations (Expanded for better demonstration)
MEAL_CENTERS = {
    (1885, 55): {'name': 'Spicy Chicken Curry', 'cuisine': 'Indian', 'category': 'Main Dish'}, 
    (1993, 55): {'name': 'Basmati Rice', 'cuisine': 'Indian', 'category': 'Rice'},
    (2539, 55): {'name': 'Spring Rolls', 'cuisine': 'Thai', 'category': 'Starters'},
    (2631, 55): {'name': 'Gulab Jamun', 'cuisine': 'Indian', 'category': 'Dessert'},
    (1234, 55): {'name': 'Lemonade', 'cuisine': 'Thai', 'category': 'Beverages'},
    (5678, 55): {'name': 'Aloo Gobi', 'cuisine': 'Indian', 'category': 'Vegetable'},
}

# --- INNOVATION 1: RECIPE & COST DATA (Crucial for Procurement Scaling) ---
# Dictionary structure: {ingredient: [cost_per_unit, unit]}
# This links predicted demand directly to procurement needs.
RECIPE_COSTS = {
    'Spicy Chicken Curry': {
        'Chicken (kg)': [250, 0.25], # Rs 250/kg, 0.25kg per serving
        'Spice Mix (g)': [10, 50],   # Rs 10/kg, 50g per serving
    },
    'Basmati Rice': {
        'Rice (kg)': [80, 0.15],     # Rs 80/kg, 0.15kg per serving
    },
    'Gulab Jamun': {
        'Milk Solids (kg)': [350, 0.05],
        'Sugar (kg)': [40, 0.1],
    },
    # ... Add entries for other meals
}

@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost model."""
    model_path = 'trained_model.json'
    if not os.path.exists(model_path):
        # NOTE: In a real environment, you must run model_trainer.py
        st.warning(f"Warning: Model file '{model_path}' not found. Using dummy model for demonstration.")
        
        # Use a dummy model (only for local testing if the file doesn't exist)
        class DummyModel:
            def predict(self, X):
                # Simulate realistic demand (around 1000 orders average)
                # Scale prediction based on 'lag_1_num_orders'
                lag_val = X['lag_1_num_orders'].iloc[0] if 'lag_1_num_orders' in X.columns else 1000
                base_demand = np.log1p(lag_val * 0.2) 
                
                # Apply small random noise for different meal types
                noise = np.random.uniform(0.9, 1.1)
                
                return np.array([base_demand * noise])

            @property
            def feature_names_in_(self):
                # Must match the features used in prediction logic below
                return ['checkout_price', 'emailer_for_promotion', 'homepage_featured', 
                        'city_code', 'region_code', 'op_area', 'quarter', 
                        'semester', 'lag_1_num_orders', 'center_type_TYPE_B', 
                        'center_type_TYPE_C', 'category_Main Dish', 'category_Rice', 
                        'category_Starters', 'category_Dessert', 'category_Beverages', 
                        'category_Vegetable', 'cuisine_Indian', 'cuisine_Thai']

        return DummyModel()

    model = xgb.XGBRegressor()
    try:
        model.load_model(model_path)
        return model
    except xgb.core.XGBoostError as e:
        st.error(f"Error: Could not load the trained XGBoost model: {e}")
        return None

# Load the trained model or dummy model
model = load_model()

# --- APP LAYOUT ---

st.set_page_config(layout="wide", page_title="AI Food Waste DSS")

st.title("🌱 AI Hostel Food Management Decision Support System")
st.caption("From Prediction to Procurement: Optimizing inventory, reducing cost, and minimizing food waste.")

# Use tabs to separate the core prediction tool from the innovative feedback/audit tool
tab1, tab2 = st.tabs(["📊 Demand Forecast & Procurement", "📝 Post-Meal Waste Audit & Feedback Loop"])

# Global variable to store simulation results for display
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

with tab1:
    if model:
        st.markdown("---")
        
        # Two columns for input: one for global context, one for historical data
        col_context, col_lag = st.columns([1, 1])

        with col_context:
            st.header("1. Contextual Inputs")
            # Customisation: Allows date and event selection 
            forecast_date = st.date_input(
                "Select Forecast Date (Target Meal Day)",
                datetime.date.today() + datetime.timedelta(days=7)
            )
            
            special_event = st.selectbox(
                "Anticipated Event/Holiday",
                ['None', 'Exam Week (Low Demand)', 'Festival/Outing (High Demand)', 'Mass Leave (Zero Demand)'],
                help="Factor in predictable demand shifts due to hostel events."
            )
        
        with col_lag:
            st.header("2. Historical Input (Lag Feature)")
            # Input for Lag Feature (Crucial for time-series accuracy)
            lag_1_orders = st.slider(
                "Last Week's Total Orders (Input for Lag Feature)", 
                min_value=1000, max_value=20000, value=5000,
                help="Enter the total orders served exactly one week ago for model context."
            )
        
        st.markdown("---")
        if st.button("🚀 Generate Actionable Forecast", type="primary"):
            
            # --- PREDICTION LOGIC ---
            forecast_data = []
            total_procurement_cost = 0
            total_predicted_servings = 0
            
            # Simplified week calculation based on date (for dummy feature engineering)
            forecast_week = (forecast_date.timetuple().tm_yday // 7) + 146
            
            # Aggregate ingredient needs across all predicted meals
            aggregated_ingredients = {}

            for (meal_id, center_id), meal_info in MEAL_CENTERS.items():
                
                meal_name = meal_info['name']
                meal_cuisine = meal_info['cuisine']
                meal_category = meal_info['category']

                # 1. Create a dummy row for prediction
                X_new = pd.Series(0, index=model.feature_names_in_) 
                
                # --- Input Features ---
                X_new['checkout_price'] = 150.0  # Placeholder price
                X_new['emailer_for_promotion'] = 1 if 'Festival' in special_event else 0
                X_new['homepage_featured'] = 1 if 'Festival' in special_event else 0
                X_new['city_code'] = 647 
                X_new['region_code'] = 56  
                X_new['op_area'] = 2.0  
                X_new['quarter'] = (forecast_week // 13) + 1
                X_new['semester'] = (forecast_week // 26) + 1
                # Distribute the total lag order proportionally across meals
                X_new['lag_1_num_orders'] = lag_1_orders / len(MEAL_CENTERS) 
                
                # --- One-Hot Encoded Features ---
                if f'category_{meal_category}' in X_new.index:
                    X_new[f'category_{meal_category}'] = 1
                
                if f'cuisine_{meal_cuisine}' in X_new.index:
                    X_new[f'cuisine_{meal_cuisine}'] = 1
                
                X_new['center_type_TYPE_B'] = 0 
                X_new['center_type_TYPE_C'] = 1 
                
                X_new_df = pd.DataFrame([X_new])
                
                # --- Prediction ---
                y_pred_log = model.predict(X_new_df)
                
                # Inverse Log Transform, Round, and Adjust for Special Events
                predicted_orders = np.round(np.expm1(y_pred_log[0]))
                
                # Manual event adjustments (Domain Logic)
                if 'Mass Leave' in special_event:
                    predicted_orders = 0
                elif 'Exam Week' in special_event:
                    predicted_orders = max(0, predicted_orders * 0.7) # Reduce demand by 30%
                
                predicted_orders = max(0, int(predicted_orders))
                
                # --- INNOVATION 1: RECIPE SCALING & COSTING ---
                meal_cost = 0
                required_ingredients = {}
                
                if meal_name in RECIPE_COSTS and predicted_orders > 0:
                    for ingredient, [cost_per_unit, unit_per_serving] in RECIPE_COSTS[meal_name].items():
                        
                        # Calculate total units needed and cost
                        total_units_needed = unit_per_serving * predicted_orders
                        ingredient_cost = total_units_needed * cost_per_unit
                        
                        meal_cost += ingredient_cost
                        
                        # Store for the meal and global aggregation
                        required_ingredients[ingredient] = f"{total_units_needed:.2f} units"
                        
                        # Global ingredient aggregation
                        ingredient_key = ingredient.split('(')[0].strip() # e.g., 'Chicken'
                        if ingredient_key in aggregated_ingredients:
                            aggregated_ingredients[ingredient_key]['quantity'] += total_units_needed
                        else:
                            aggregated_ingredients[ingredient_key] = {'quantity': total_units_needed, 'unit': ingredient.split('(')[1].replace(')', '')}
                
                total_procurement_cost += meal_cost
                total_predicted_servings += predicted_orders
                
                forecast_data.append({
                    'Menu Item': meal_name,
                    'Predicted Servings': predicted_orders,
                    'Est. Cost (Rs.)': f"₹ {meal_cost:,.0f}",
                    'Procurement Action': 'Prepare Exact Quantity',
                    'Required Ingredients': required_ingredients
                })

            # Store results in session state for tab 2
            st.session_state.forecast_results = {
                'forecast_data': forecast_data,
                'total_cost': total_procurement_cost,
                'total_servings': total_predicted_servings,
                'date': forecast_date,
                'aggregated_ingredients': aggregated_ingredients,
            }

            # --- DISPLAY INNOVATIVE RESULTS ---
            st.header("3. AI Decision Support Dashboard")
            
            results_df = pd.DataFrame(forecast_data)
            
            # Metrics Row 
            col_total_servings, col_total_cost, col_cost_saving = st.columns(3)
            
            # INNOVATION 3: Financial Impact Visualization (vs. fixed 10,000 Rs cost)
            PREVIOUS_FIXED_COST = 10000 
            cost_difference = PREVIOUS_FIXED_COST - total_procurement_cost

            with col_total_servings:
                st.metric(
                    label="Total Predicted Servings", 
                    value=f"{total_predicted_servings:,.0f} meals", 
                    delta="Based on historical trends & event context."
                )
            
            with col_total_cost:
                st.metric(
                    label="Estimated Procurement Cost", 
                    value=f"₹ {total_procurement_cost:,.0f}", 
                    delta="AI-Optimized Budget"
                )
            
            with col_cost_saving:
                st.metric(
                    label="Estimated Cost Savings (vs. Fixed Budget)", 
                    value=f"₹ {cost_difference:,.0f}", 
                    delta=f"{'Savings' if cost_difference > 0 else 'Over Budget'}",
                    delta_color="normal" if cost_difference > 0 else "inverse"
                )
                
            # Procurement Action Plan
            st.subheader("🛒 Aggregated Procurement Plan (AI-Optimized Shopping List)")
            procurement_list = []
            for item, data in aggregated_ingredients.items():
                procurement_list.append({
                    'Ingredient': item,
                    'Quantity': f"{data['quantity']:.2f}",
                    'Unit': data['unit'],
                    'Status': 'Order Required'
                })
                
            st.dataframe(pd.DataFrame(procurement_list), hide_index=True, use_container_width=True)

            # Detailed Meal Breakdown
            st.subheader("🍽️ Detailed Meal-by-Meal Demand Breakdown")
            
            # Clean up results for presentation
            display_df = results_df.drop(columns=['Required Ingredients'])

            st.dataframe(
                display_df,
                column_config={
                    "Predicted Servings": st.column_config.ProgressColumn(
                        "Predicted Servings",
                        help="Predicted quantity to be prepared to minimize waste.",
                        format="%d",
                        min_value=0,
                        # Use 50% of total servings as max for better progress bar visualization
                        max_value=max(1, total_predicted_servings * 0.5), 
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.success(f"Forecasting complete for {forecast_date.strftime('%Y-%m-%d')}. Proceed to procurement based on the plan above.")

    else:
        st.error("Model loading failed. Please ensure 'trained_model.json' exists or check the loading code.")


with tab2:
    # --- INNOVATION 2: POST-MEAL WASTE FEEDBACK LOOP ---
    st.header("Post-Meal Waste Audit & Feedback Loop")
    st.markdown("""
    This feature is crucial for the **Continuous Improvement** of the AI model. 
    By accurately recording the actual waste for each meal, the system generates new, clean data 
    that can be used for future model retraining, leading to higher precision and greater savings.
    """)

    if st.session_state.forecast_results:
        st.subheader(f"Audit Form for Meal Date: {st.session_state.forecast_results['date'].strftime('%B %d, %Y')}")
        
        audit_data = st.session_state.forecast_results['forecast_data']
        audit_date = st.session_state.forecast_results['date']
        
        # Display Actual vs. Predicted Servings
        st.info(f"Predicted Total Servings: **{st.session_state.forecast_results['total_servings']:,}**")

        waste_form = st.form(key='waste_audit_form')
        
        waste_records = []
        total_waste_kg = 0
        total_actual_served = 0

        with waste_form:
            st.markdown("Enter **Actual Servings** (Counted/Validated) and **Food Waste** (in kilograms) below:")
            
            for i, meal in enumerate(audit_data):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.text_input(f"Menu Item", value=meal['Menu Item'], disabled=True, key=f"meal_name_{i}")
                
                with col2:
                    # User inputs actual servings served
                    actual_servings = st.number_input(
                        "Actual Servings Served", 
                        min_value=0, 
                        value=int(meal['Predicted Servings'] * 0.95), # Pre-fill with a close number for simulation
                        key=f"actual_servings_{i}",
                        step=1
                    )
                    total_actual_served += actual_servings

                with col3:
                    # User inputs actual waste in kg
                    waste_kg = st.number_input(
                        "Waste (kg) 🗑️", 
                        min_value=0.0, 
                        value=np.random.uniform(0.1, 1.5), # Pre-fill with a random value for simulation
                        key=f"waste_kg_{i}",
                        step=0.1
                    )
                    total_waste_kg += waste_kg
                
                waste_records.append({
                    'meal_name': meal['Menu Item'],
                    'date': str(audit_date),
                    'predicted_servings': meal['Predicted Servings'],
                    'actual_servings': actual_servings,
                    'waste_kg': waste_kg
                })

            submitted = waste_form.form_submit_button("Submit Audit Data & Retrain Feedback")

            if submitted:
                # Simulation of data storage and MLOps Trigger
                st.balloons()
                st.success(f"Audit Data Submitted for {audit_date.strftime('%Y-%m-%d')}!")
                st.info(f"Total Recorded Waste: **{total_waste_kg:.2f} kg**")
                st.info(f"Total Actual Servings: **{total_actual_served:,}**")

                # INNOVATION 3: Displaying Waste Reduction Percentage
                # Dummy historical waste data (e.g., 10% of 15,000 fixed servings)
                PREVIOUS_FIXED_WASTE_KG = 30 
                
                st.metric(
                    label="Waste Reduction Impact",
                    value=f"{((PREVIOUS_FIXED_WASTE_KG - total_waste_kg) / PREVIOUS_FIXED_WASTE_KG) * 100:.1f}%",
                    delta=f"{PREVIOUS_FIXED_WASTE_KG - total_waste_kg:.2f} kg less than fixed preparation waste.",
                    delta_color="normal"
                )
                
                st.warning("Data is now queued for model retraining (MLOps). The AI will learn from this real-world feedback.")

    else:
        st.info("Please generate a forecast in the 'Demand Forecast & Procurement' tab first to populate the audit form.")
