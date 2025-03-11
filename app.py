import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
# Load Data
@st.cache_data
def load_data():
    file_path = "renamed_data.xlsx"
    xls = pd.ExcelFile(file_path)
    df_list = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df["WAREHOUSE"] = sheet  # Assign warehouse name from sheet
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors='coerce')
    return df_all

df = load_data()
# Sidebar Navigation with Hover Effect
st.markdown(
    """
    <style>
        .sidebar-nav {
            font-size: 18px;
        }
        .sidebar-nav a {
            display: block;
            padding: 10px;
            text-decoration: none;
            color: white;
            background: #333;
            border-radius: 5px;
            margin-bottom: 5px;
            transition: background 0.3s ease-in-out;
        }
        .sidebar-nav a:hover {
            background: #D7A6FF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# with st.sidebar:
#     st.markdown("### 🌍 Dashboard Navigation", unsafe_allow_html=True)
#     selected_page = st.markdown(
#         """
#         <div class="sidebar-nav">
#             <a href="?page=productivity">📊 Productivity Overview</a>
#             <a href="?page=weather">🌡️ Weather Overview</a>
#             <a href="?page=resource">👷 Resource Overview</a>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # Get page from query params
# query_params = st.experimental_get_query_params()
# selected_page = query_params.get("page", ["weather"])[0]
with st.sidebar:
    st.markdown("### 🌍 Dashboard Navigation", unsafe_allow_html=True)
    selected_pagee = st.markdown(
        """
        <div class="sidebar-nav">
            <a href='?page=productivity'>📊 Productivity Overview</a>
            <a href='?page=weather'>🌡️ Weather Overview</a>
            <a href='?page=resource'>👷 Resource Overview</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Get page from query params
query_params = st.query_params
if "page" in query_params:
    st.session_state.selected_page = query_params["page"]
else:
    st.session_state.selected_page = "productivity"
selected_page = st.session_state.selected_page
if selected_page == "productivity":
    st.title("📊 Productivity Overview")

    # Filters at the Top
    st.markdown("### Filters")
    col1, col2 = st.columns(2)
    with col1:
        selected_warehouse = st.selectbox("Select Warehouse", ["All"] + list(df["WAREHOUSE"].unique()))
    with col2:
        selected_date = st.date_input("Select Date", df["Date"].min())

    # Apply Filters
    if selected_warehouse != "All":
        df = df[df["WAREHOUSE"] == selected_warehouse]
    df = df[df["Date"] == pd.to_datetime(selected_date)]

    # KPIs Section in Pastel Boxes
    avg_steps = df["EstimatedStepsTaken"].mean()
    avg_pick_distance = df["EstimatedPickDistanceMeters"].mean()
    avg_processing_time = df["TimeDiffSeconds"].mean()
    total_picked = df["TotalPickedLines"].sum()
    total_lines = df["TotalProcessedQty"].sum()
    progress_percentage = total_picked / total_lines if total_lines > 0 else 0

    # kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    # with kpi_col1:
    #     st.markdown(
    #         f"""
    #         <div style='background:#FFDDC1; padding:10px; border-radius:10px; text-align:center;'>
    #             <h4>Avg Steps Taken</h4>
    #             <h2>{avg_steps:.2f}</h2>
    #         </div>
    #         """, unsafe_allow_html=True
    #     )
    # with kpi_col2:
    #     st.markdown(
    #         f"""
    #         <div style='background:#C1E1FF; padding:10px; border-radius:10px; text-align:center;'>
    #             <h4>Avg Pick Distance (m)</h4>
    #             <h2>{avg_pick_distance:.2f}</h2>
    #         </div>
    #         """, unsafe_allow_html=True
    #     )
    # with kpi_col3:
    #     st.markdown(
    #         f"""
    #         <div style='background:#D4FAC1; padding:10px; border-radius:10px; text-align:center;'>
    #             <h4>Avg Processing Time (s)</h4>
    #             <h2>{avg_processing_time:.2f}</h2>
    #         </div>
    #         """, unsafe_allow_html=True
    #     )
    avg_items_per_hour = df["TotalProcessedQty"].sum() / (df["TimeDiffSeconds"].sum() / 3600) if df["TimeDiffSeconds"].sum() > 0 else 0
    avg_weight_per_order = df["TotalWeightProcessedKg"].sum() / df["TotalProcessedQty"].sum() if df["TotalProcessedQty"].sum() > 0 else 0
    avg_orders_per_worker = df["TotalProcessedQty"].sum() / df["RESOURCE"].nunique() if df["RESOURCE"].nunique() > 0 else 0

    total_weight_processed = df["TotalWeightProcessedKg"].sum()
    total_orders_processed = df["TotalProcessedQty"].sum()

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    kpi_style = "background:#FFC3A0; padding:10px; border-radius:15px; text-align:center; width:100%; height:160px; display:flex; flex-direction:column; justify-content:center; align-items:center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);"

    with kpi_col1:
        st.markdown(
            f"""
            <div style='{kpi_style.replace("#FFC3A0", "#F1F8E4")}'>
                <h4 style='margin:0;'>Avg Items per Hour</h4>
                <h2 style='margin:5px 0;'>{avg_items_per_hour:.2f}</h2>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div style='{kpi_style.replace("#FFC3A0", "#E0BBE4")}'>
                <h4 style='margin:0;'>Avg Weight per Order</h4>
                <h2 style='margin:5px 0;'>{avg_weight_per_order:.2f} kg</h2>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div style='{kpi_style.replace("#FFC3A0", "#FFD3B6")}'>
                <h4 style='margin:0;'>Avg Orders per Worker</h4>
                <h2 style='margin:5px 0;'>{avg_orders_per_worker:.2f}</h2>
            </div>
            """, unsafe_allow_html=True
        )
    # Enhanced Progress Bar with Better Styling - Removed boxes below
    st.markdown("### Progress: Picked Lines vs Total Lines")
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="flex-grow: 1; background-color: #f0f2f6; border-radius: 10px; height: 30px; position: relative; overflow: hidden;">
                    <div style="position: absolute; height: 100%; width: {progress_percentage:.1%}; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px;">
                    </div>
                </div>
                <div style="margin-left: 10px; font-size: 18px; font-weight: bold; min-width: 60px; text-align: right;">
                    {progress_percentage:.1%}
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; color: #586069; font-size: 14px;">
                <div>0 of {total_lines} lines</div>
                <div>{total_picked} of {total_lines} lines</div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Layout for Original Charts
    chart_col1, chart_col2 = st.columns(2)

    # Hourly Productivity (Line Chart)
    with chart_col1:
        hourly_trend = df.groupby("Hour")["TotalProcessedQty"].sum().reset_index()
        fig1 = px.line(hourly_trend, x="Hour", y="TotalProcessedQty", title="Hourly Productivity", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    # Replace with Weight Analysis by Region
   # Calculate the average weight per item per region
    
    # # Replace with Weight Analysis by Region
    with chart_col2:
        # Calculate average item weight
        df['AvgItemWeight'] = df['TotalWeightProcessedKg'].mean()
        
        # Group by region
        weight_by_region = df.groupby('REGION')[['AvgItemWeight', 'TotalWeightProcessedKg']].mean().reset_index()
        
        # Create dual-axis chart
        fig_weight = px.bar(weight_by_region, x="REGION", y="TotalWeightProcessedKg",
                        title="Weight Analysis by Region",
                        labels={"TotalWeightProcessedKg": "Total Weight (kg)", 
                                "REGION": "Region", 
                                "AvgItemWeight": "Avg Item Weight (kg)"},
                        template="plotly_dark")
        
        # Add average item weight as a line on secondary y-axis
        fig_weight.add_trace(
            px.line(weight_by_region, x="REGION", y="AvgItemWeight").data[0]
        )
        
        # Update secondary y-axis
        fig_weight.update_traces(yaxis="y2", selector=dict(type='scatter'))
        fig_weight.update_layout(
            yaxis2=dict(
                title="Avg Item Weight (kg)",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_weight, use_container_width=True)

    # Another row for original charts
    chart_col3, chart_col4 = st.columns(2)

    # Region-wise Productivity (Donut Chart)
    with chart_col3:
        region_productivity = df.groupby("REGION")["TotalProcessedQty"].sum().reset_index()
        fig3 = px.pie(region_productivity, names="REGION", values="TotalProcessedQty", title="Productivity by Region", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # Steps Taken vs Productivity (Scatter Plot)
    with chart_col4:
        fig4 = px.scatter(df, x="EstimatedStepsTaken", y="TotalProcessedQty", title="Steps Taken vs Productivity", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)


    chart_col5, chart_col6 = st.columns(2)

    # Box Plot of Processing Times by Region
    with chart_col5:
        fig6 = px.box(
            df, 
            x="REGION", 
            y="TimeDiffSeconds",
            title="Processing Time Distribution by Region",
            color="REGION",
            template="plotly_dark"
        )
        
        fig6.update_layout(
            xaxis_title="Region",
            yaxis_title="Processing Time (seconds)",
            showlegend=False
        )
        
        st.plotly_chart(fig6, use_container_width=True)

    # Efficiency Metric (Items Processed per Step) - Reformatted
    # with chart_col6:
    #     # Calculate efficiency (items per step)
    #     df['EfficiencyRatio'] = df['TotalProcessedQty'] / df['EstimatedStepsTaken'].where(df['EstimatedStepsTaken'] > 0, 1)
        
    #     # Group by region
    #     efficiency_by_region = df.groupby('REGION')['EfficiencyRatio'].mean().reset_index()
    #     efficiency_by_region = efficiency_by_region.sort_values('EfficiencyRatio', ascending=False)
        
    #     # Create horizontal bar chart with improved formatting
    #     fig8 = px.bar(
    #         efficiency_by_region,
    #         y='REGION',
    #         x='EfficiencyRatio',
    #         title="Efficiency Ratio by Region",
    #         orientation='h',
    #         color='EfficiencyRatio',
    #         color_continuous_scale='Viridis',
    #         template="plotly_dark",
    #         text=efficiency_by_region['EfficiencyRatio'].apply(lambda x: f"{x:.2f}")  # Add text labels
    #     )
        
    #     # Improved layout and formatting
    #     fig8.update_layout(
    #         yaxis_title="Region",
    #         xaxis_title="Items Processed per Step",
    #         height=400,
    #         margin=dict(l=20, r=20, t=40, b=20),
    #         coloraxis_showscale=False  # Hide the color scale
    #     )
        
    #     # Improve text positioning and formatting
    #     fig8.update_traces(
    #         textposition='outside',
    #         textfont=dict(size=12, color='white')
    #     )
        
    #     st.plotly_chart(fig8, use_container_width=True)
    with chart_col6:
        pick_volume = df.groupby("REGION")["TotalPickedLines"].sum().reset_index()

# Create bar chart using Plotly
        fig4 = px.bar(pick_volume, x='REGION', y='TotalPickedLines',
              title="Pick Volume Analysis by Region",
              labels={'TotalPickedLines': 'Total Picked Lines'},
              color='REGION', color_discrete_sequence=px.colors.sequential.Viridis, template="plotly_white")

        fig4.update_layout(yaxis_title="Total Picked Lines", xaxis_title="Region", showlegend=False)

# Display in Streamlit
        st.plotly_chart(fig4, use_container_width=True)
        # df['EfficiencyRatio'] = df['TotalProcessedQty'] / df['EstimatedStepsTaken'].where(df['EstimatedStepsTaken'] > 0, 1)
        # efficiency_by_region = df.groupby('REGION')['EfficiencyRatio'].mean().reset_index()
        
        # fig4 = px.bar(efficiency_by_region, x='REGION', y='EfficiencyRatio',
        #             title="Efficiency Ratio by Region",
        #             labels={'EfficiencyRatio': 'Items Processed per Step'},
        #             color='REGION', color_discrete_sequence=px.colors.sequential.Plasma, template="plotly_white")
        
        # fig4.update_layout(yaxis_title="Efficiency Ratio", xaxis_title="Region", showlegend=False)
        
        # st.plotly_chart(fig4, use_container_width=True)
elif selected_page == "weather":
    st.title("🌡️ Weather Overview")
    
    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_warehouse = st.selectbox("🏢 Select Warehouse", ["All"] + list(df["WAREHOUSE"].unique()))
    with col2:
        selected_date = st.date_input("📅 Select Date", df["Date"].min())
    with col3:
        temp_range = st.slider("🌡️ Select Temperature Range", int(df["Temperature"].min()), int(df["Temperature"].max()), (int(df["Temperature"].min()), int(df["Temperature"].max())))
    
    df_filtered = df.copy()
    if selected_warehouse != "All":
        df_filtered = df_filtered[df_filtered["WAREHOUSE"] == selected_warehouse]
    df_filtered = df_filtered[(df_filtered["Date"] == pd.to_datetime(selected_date)) & (df_filtered["Temperature"].between(temp_range[0], temp_range[1]))]
    
    # Create Temperature Bins
    temp_min = df_filtered["Temperature"].min()
    temp_max = df_filtered["Temperature"].max()
    bins = [temp_min, temp_min + (temp_max-temp_min)*0.2, temp_min + (temp_max-temp_min)*0.4,
            temp_min + (temp_max-temp_min)*0.6, temp_min + (temp_max-temp_min)*0.8, temp_max]
    labels = ["Very Cold", "Cold", "Mild", "Warm", "Hot"]
    df_filtered["Temperature Category"] = pd.cut(df_filtered["Temperature"], bins=bins, labels=labels, include_lowest=True)
    
    # Scatter Plots Side by Side
    #st.markdown("### Temperature & Humidity Impact on Productivity")
    scatter_col1, scatter_col2 = st.columns(2)
    
    # with scatter_col1:
    #     # Removed color="Temperature Category"
    #     fig1 = px.scatter(df_filtered, x="Temperature", y="TotalProcessedQty", trendline="ols", title="Temperature vs Productivity", template="plotly_dark", width=900)
    #     fig1.update_traces(marker=dict(size=10, opacity=0.7))
    #     st.plotly_chart(fig1, use_container_width=True)
    with scatter_col1:
    # Create a copy with rounded temperatures
        df_temp_rounded = df_filtered.copy()
        df_temp_rounded["Temperature"] = df_temp_rounded["Temperature"].round(0)
        
        # Create scatter plot with red trendline
        fig1 = px.scatter(df_temp_rounded, x="Temperature", y="TotalProcessedQty", 
                        trendline="ols", title="Temperature vs Productivity", 
                        template="plotly_dark", width=900)
        
        # Update marker properties
        fig1.update_traces(marker=dict(size=10, opacity=0.7))
        
        # Update trendline color to red
        for trace in fig1.data:
            if trace.mode == 'lines':
                trace.line.color = 'red'
                trace.line.width = 3
        
        st.plotly_chart(fig1, use_container_width=True)
    with scatter_col2:
        # Removed color="Temperature Category" and added trendline="ols"
        fig2 = px.scatter(df_filtered, x="Humidity", y="TotalProcessedQty", title="Humidity vs Productivity", template="plotly_dark", width=900)
        fig2.update_traces(marker=dict(size=10, opacity=0.7))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Temperature & Humidity Fluctuation in a Single Line Chart
    #st.markdown("### Temperature & Humidity Fluctuation by Hour")
    temp_humid_hour = df_filtered.groupby("Hour")[["Temperature", "Humidity"]].mean().reset_index()
    fig3 = px.line(temp_humid_hour, x="Hour", y=["Temperature", "Humidity"], title="Temperature & Humidity Trends",
                   labels={"value": "Measurement", "variable": "Metric"}, template="plotly_dark")
    fig3.update_traces(mode="lines+markers")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Bar Chart for Temperature Categories vs Productivity & Pie Chart Side by Side
    #st.markdown("### Temperature Category Impact on Productivity")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        temp_cat_productivity = df_filtered.groupby("Temperature Category")["TotalProcessedQty"].sum().reset_index()
        fig4 = px.bar(temp_cat_productivity, x="Temperature Category", y="TotalProcessedQty",
                      title="Productivity by Temperature Category", template="plotly_dark", color="Temperature Category")
        st.plotly_chart(fig4, use_container_width=True)
    
    with chart_col2:
        fig5 = px.pie(df_filtered, names="Temperature Category", title="Temperature Distribution", template="plotly_dark")
        st.plotly_chart(fig5, use_container_width=True)
if selected_page == "resource":
    st.title("Resource Performance Dashboard")
    
    # Filters - Add warehouse filter and make "All" an option for both filters
    st.markdown("### Select Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_warehouse = st.selectbox("🏢 Select Warehouse", ["All"] + list(df["WAREHOUSE"].unique()))
    with col2:
        # Filter resources based on warehouse selection
        if selected_warehouse == "All":
            resources_list = ["All"] + list(df["RESOURCE"].unique())
        else:
            resources_list = ["All"] + list(df[df["WAREHOUSE"] == selected_warehouse]["RESOURCE"].unique())
        selected_resource = st.selectbox("🆔 Select Resource", resources_list)
    with col3:
        date_range = st.date_input("📅 Date Range", 
                                   [df["Date"].min(), df["Date"].max()],
                                   min_value=df["Date"].min(),
                                   max_value=df["Date"].max())
    
    # Filter data based on selections
    if selected_warehouse != "All":
        df_filtered = df[df["WAREHOUSE"] == selected_warehouse].copy()
    else:
        df_filtered = df.copy()
        
    if selected_resource != "All":
        df_resource = df_filtered[df_filtered["RESOURCE"] == selected_resource].copy()
    else:
        df_resource = df_filtered.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_resource = df_resource[(df_resource["Date"] >= pd.to_datetime(start_date)) & 
                                  (df_resource["Date"] <= pd.to_datetime(end_date))]
    
    # Compute performance metrics
    df_resource["Quantity Performance %"] = (df_resource["TotalProcessedQty"] / (df_resource["target_qty_rateperhour"] * df_resource["TimeDiffSeconds"] / 3600)) * 100
    df_resource["Completed vs Open Ratio"] = df_resource["TotalPickedLines"] / (df_resource["TotalOpenLines"] + 1)
    
    # Resource KPIs
    st.markdown("### 📊 Performance Overview")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    avg_qty_perf = df_resource["Quantity Performance %"].mean()
    total_qty = df_resource["TotalProcessedQty"].sum()
    total_weight = df_resource["TotalWeightProcessedKg"].sum()
    
    kpi_style = "background:#F0F8FF; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
    
    with kpi_col1:
        st.markdown(f"<div style='{kpi_style}'><h4>📈 Quantity Performance</h4><h2>{avg_qty_perf:.1f}%</h2></div>", unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"<div style='{kpi_style.replace('#F0F8FF', '#E6F3E6')}'><h4>📦 Total Items Processed</h4><h2>{total_qty}</h2></div>", unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"<div style='{kpi_style.replace('#F0F8FF', '#FFF0F5')}'><h4>⚖️ Total Weight Processed</h4><h2>{total_weight:.1f} kg</h2></div>", unsafe_allow_html=True)
    
    # Daily performance trends
    st.markdown("### 📈 Daily Performance Trends")
    daily_perf = df_resource.groupby("Date")[["TotalProcessedQty", "TotalWeightProcessedKg"]].mean().reset_index()
    fig1 = px.line(daily_perf, x="Date", y=["TotalProcessedQty"], title="Daily Performance Trends", template="plotly_white", markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # New Chart: Temperature Impact on Worker Performance
    st.markdown("### ☀️ Temperature Impact on Worker Performance")
    fig2 = px.scatter(df_resource, x="Temperature", y="Quantity Performance %", color="RESOURCE", title="Effect of Temperature on Worker Performance", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    
    # New Chart: Worker Performance by Region
    st.markdown("### 🏢 Worker Performance by Region")
    resource_region_perf = df_resource.groupby(["REGION", "RESOURCE"])["Quantity Performance %"].mean().reset_index()
    fig3 = px.bar(resource_region_perf, x="REGION", y="Quantity Performance %", color="RESOURCE", barmode="group", 
                  title="Resource Performance Across Regions", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    
    # st.markdown("### Shift-wise Productivity Heatmap")
    # pivot_table = df_resource.pivot_table(index="Hour", columns="RESOURCE", values="Quantity Performance %", aggfunc="mean")
    # fig5, ax = plt.subplots(figsize=(12, 6))
    # sns.heatmap(pivot_table, cmap="coolwarm", annot=False, fmt=".1f", linewidths=0.5, ax=ax)
    # plt.title("Productivity Heatmap by Hour and Resource")
    # st.pyplot(fig5)
    st.markdown("### Shift-wise Productivity Heatmap")

    # Create pivot table for heatmap
    pivot_table = df_resource.pivot_table(index="Hour", columns="RESOURCE", values="Quantity Performance %", aggfunc="mean")

    # Create figure and axis
    fig5, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap with custom color map (green for high values, light red for low)
    sns.heatmap(
        pivot_table, 
        cmap="RdYlGn",  # Red-Yellow-Green colormap (red for low, green for high)
        annot=True,     # Show values in cells
        fmt=".1f",      # Format as single decimal 
        linewidths=0.5,
        ax=ax,
        vmin=pivot_table.values.min(),  # Set color scale minimum
        vmax=pivot_table.values.max(),  # Set color scale maximum
        center=pivot_table.values.mean()  # Center color scale at the mean value
    )

    # Add title and labels
    plt.title("Productivity Heatmap by Hour and Resource", fontsize=14)
    plt.xlabel("Resource", fontsize=12)
    plt.ylabel("Hour", fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig5)
        
    # Summary Table
    st.markdown("### 📊 Daily Performance Summary")
    st.dataframe(daily_perf, use_container_width=True)