import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("📊 Data Analysis Dashboard")
st.markdown("Interactive dashboard for data analysis and visualization")

# Load the data


@st.cache_data
def load_data():
    df = pd.read_csv('dataset_final.csv')
    return df


try:
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Add date filter if there's a date column
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    if date_columns:
        selected_date_column = st.sidebar.selectbox(
            "Select Date Column", date_columns)
        df[selected_date_column] = pd.to_datetime(df[selected_date_column])
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [df[selected_date_column].min(), df[selected_date_column].max()]
        )

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Statistics"])

    with tab1:
        st.header("Data Overview")

        # Display basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")

        with col2:
            st.subheader("Column Names")
            st.write(df.columns.tolist())

        # Display first few rows
        st.subheader("Preview of Data")
        st.dataframe(df.head())

        # Display missing values
        st.subheader("Missing Values Analysis")
        missing_values = df.isnull().sum()
        fig = px.bar(x=missing_values.index, y=missing_values.values,
                     title="Missing Values by Column")
        st.plotly_chart(fig)

    with tab2:
        st.header("Detailed Analysis")

        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        if len(numeric_columns) > 0:
            st.subheader("Numeric Data Analysis")
            selected_numeric = st.selectbox(
                "Select Numeric Column", numeric_columns)

            # Create histogram
            fig = px.histogram(df, x=selected_numeric,
                               title=f"Distribution of {selected_numeric}")
            st.plotly_chart(fig)

            # Create box plot
            fig = px.box(df, y=selected_numeric,
                         title=f"Box Plot of {selected_numeric}")
            st.plotly_chart(fig)

        if len(categorical_columns) > 0:
            st.subheader("Categorical Data Analysis")
            selected_categorical = st.selectbox(
                "Select Categorical Column", categorical_columns)

            # Create bar chart
            value_counts = df[selected_categorical].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                         title=f"Distribution of {selected_categorical}")
            st.plotly_chart(fig)

    with tab3:
        st.header("Statistical Analysis")

        # Display correlation matrix for numeric columns
        if len(numeric_columns) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix,
                            title="Correlation Matrix",
                            labels=dict(color="Correlation"))
            st.plotly_chart(fig)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure the dataset file is in the correct location and format.")
