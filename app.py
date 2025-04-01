import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_object_dtype, is_categorical_dtype
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings

def inject_dark_theme_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Set page config
st.set_page_config(
    page_title="Smart Data Prep & EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)


inject_dark_theme_css()

# App Title & Intro

st.title("Smart Data Pre-Processing & Exploratory Data Analysis")



st.markdown(
    """
    **Load. Clean. Explore. No Code Needed.** 

    Whether you're prepping data for machine learning or just making sense of a new dataset — this app gives you an intuitive, powerful interface to handle it all.

    **Clean. Format. Explore. Prep — in one seamless flow:**
    - Smart data cleaning  
    - Auto date handling  
    - Visual insights  
    - Encoding & scaling made simple
    
    _Give it a spin — your data deserves better._ 

    """
)

st.markdown("---")


# File uploader
uploaded_file = st.file_uploader("Upload your CSV file to get started", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Initial dtype conversion
    df = df.convert_dtypes()
    df = df.infer_objects()

    # Create cleaned copy early so all steps can use it
    df_cleaned = df.copy()

    # -------------------------------
    # 1️⃣ Data Preview
    # -------------------------------
    st.subheader("Data Preview")

    col1, col2 = st.columns([6, 1])  # Wide space on left, narrow on right


    with col2:
        show_full_data = st.checkbox("Show full dataset", value=False)

    if show_full_data:
        st.dataframe(df)
    else:
        st.dataframe(df.head(50))



    # Dataset shape
    st.write(f"**Shape of dataset:** {df.shape[0]} rows × {df.shape[1]} columns")


    # -------------------------------
    # 🧼 Data Cleaning Section
    # -------------------------------

    st.markdown("### Data Cleaning")
    with st.expander("Data Cleaning Tasks"):
        cleaning_task = st.selectbox("Select a cleaning task:", [
            "No Action",
            "Remove Duplicate Rows",
            "Drop Unnecessary Columns",
            "Rename Columns",
        ])

        if cleaning_task == "Remove Duplicate Rows":
            total_dupes = df_cleaned.duplicated().sum()
            if total_dupes == 0:
                st.success("No exact duplicate rows found.")
            else:
                st.warning(f"Found {total_dupes} exact duplicate rows.")
                with st.expander("Preview duplicate rows"):
                    st.dataframe(df_cleaned[df_cleaned.duplicated(keep=False)].head(10))

                dup_mode = st.radio("Method:", [
                    "Remove exact duplicates (all columns)",
                    "Remove based on specific columns"
                ])

                keep_option = st.selectbox("Keep which duplicate?", ["first", "last", "none"])
                keep_value = None if keep_option == "none" else keep_option

                if dup_mode == "Remove exact duplicates (all columns)":
                    if st.button("Remove Duplicates"):
                        df_cleaned.drop_duplicates(keep=keep_value, inplace=True)
                        st.success("Duplicates removed.")
                else:
                    subset_cols = st.multiselect("Select columns to define duplicates:", df_cleaned.columns)
                    if st.button("Remove Duplicates by Subset") and subset_cols:
                        before = df_cleaned.shape[0]
                        df_cleaned.drop_duplicates(subset=subset_cols, keep=keep_value, inplace=True)
                        after = df_cleaned.shape[0]
                        st.success(f"Removed {before - after} duplicates based on selected columns.")

        elif cleaning_task == "Drop Unnecessary Columns":
            cols_to_drop = st.multiselect("Select columns to drop:", df_cleaned.columns)
            if cols_to_drop:
                if st.button("Drop Selected Columns"):
                    df_cleaned.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"Dropped: {', '.join(cols_to_drop)}")

        elif cleaning_task == "Rename Columns":
            st.markdown("Edit the column names below:")
            rename_map = {}
            for col in df_cleaned.columns:
                new_name = st.text_input(f"Rename `{col}` to:", value=col, key=f"rename_{col}")
                if new_name and new_name != col:
                    rename_map[col] = new_name
            if rename_map:
                if st.button("Apply Renaming"):
                    df_cleaned.rename(columns=rename_map, inplace=True)
                    st.success("Renaming applied.")

    
    # Show top 50 cleaned rows after all cleaning operations
    st.markdown("#### Data Preview After Cleaning")
    st.dataframe(df_cleaned.head(50))

    # Save cleaned DataFrame for future steps
    st.session_state.cleaned_df = df_cleaned.copy()


    # -------------------------------
    # 2️⃣ Dataset Summary
    # -------------------------------
    st.subheader("Dataset Summary")

    # Use cleaned data from session state if available
    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    # Column types
    st.dataframe(df_cleaned.dtypes.astype(str).rename("Data Type"))

    # Missing values
    st.markdown("#### Missing Values")

    missing = df_cleaned.isnull().sum()
    missing = missing[missing > 0]  # Only show columns with missing values

    if not missing.empty:
        st.dataframe(missing.rename("Missing Values"))
    else:
        st.success("No missing values found in the dataset.")

    # Categorical and Numerical feature breakdown
    st.markdown("#### Column Breakdown")

    cat_cols = [
        col for col in df_cleaned.columns
        if is_object_dtype(df_cleaned[col]) 
        or isinstance(df_cleaned[col].dtype, pd.CategoricalDtype)
        or df_cleaned[col].dtype.name == "string"]

    num_cols = [
        col for col in df_cleaned.columns
        if pd.api.types.is_numeric_dtype(df_cleaned[col])]

    st.write(f"**Numerical Columns ({len(num_cols)}):**", num_cols)
    st.write(f"**Categorical Columns ({len(cat_cols)}):**", cat_cols)

    # Descriptive statistics
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_cleaned.describe().astype(str))


    # -------------------------------
    # 🗓️ Date/Time Formatting
    # -------------------------------
    st.markdown("#### Date/Time Column Detection & Conversion")

    # Load cleaned dataset from session state
    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    # Step 1: Detect potential date-like columns
    potential_dates = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ["object", "string"]]

    date_candidates = []
    for col in potential_dates:
        try:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df_cleaned[col], errors="coerce")
            if parsed.notna().sum() > 0:
                date_candidates.append(col)
        except Exception:
            continue

    if not date_candidates:
        st.info("No potential datetime columns detected.")
    else:
        selected_date_cols = st.multiselect("Select columns to convert to datetime:", date_candidates)

        for col in selected_date_cols:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            parsed = pd.to_datetime(df_cleaned[col], errors="coerce")
            failure_rate = parsed.isna().mean()

            if failure_rate > 0.3:
                st.warning(f"⚠️ `{col}` could not be consistently parsed — contains mixed or invalid formats.")
            else:
                df_cleaned[col] = parsed.dt.strftime("%Y-%m-%d")
                st.success(f"`{col}` converted and standardized to YYYY-MM-DD format.")

        # Optional: Extract components
        if selected_date_cols:
            st.markdown("#### Extract features from datetime column")
            col_to_extract = st.selectbox("Select a datetime column to extract from:", selected_date_cols)

            df_cleaned[col_to_extract] = pd.to_datetime(df_cleaned[col_to_extract], errors="coerce")

            if pd.api.types.is_datetime64_any_dtype(df_cleaned[col_to_extract]):
                with st.expander("Select features to extract"):
                    extract_year = st.checkbox("Year")
                    extract_month = st.checkbox("Month")
                    extract_day = st.checkbox("Day")
                    extract_dayofweek = st.checkbox("Day of Week")
                    extract_hour = st.checkbox("Hour")

                if extract_year:
                    df_cleaned[f"{col_to_extract}_Year"] = df_cleaned[col_to_extract].dt.year
                if extract_month:
                    df_cleaned[f"{col_to_extract}_Month"] = df_cleaned[col_to_extract].dt.month
                if extract_day:
                    df_cleaned[f"{col_to_extract}_Day"] = df_cleaned[col_to_extract].dt.day
                if extract_dayofweek:
                    df_cleaned[f"{col_to_extract}_Weekday"] = df_cleaned[col_to_extract].dt.day_name()
                if extract_hour:
                    df_cleaned[f"{col_to_extract}_Hour"] = df_cleaned[col_to_extract].dt.hour

                st.success("Date parts extracted.")
                st.markdown("#### Preview After Extraction")
                st.dataframe(df_cleaned.head())

    # Save updates back to session state
    st.session_state.cleaned_df = df_cleaned.copy()

    # -------------------------------
    # 📊 Correlation Heatmap
    # -------------------------------
    st.markdown("#### Correlation Heatmap (Numerical Features Only)")

    num_cols = [col for col in df_cleaned.columns if pd.api.types.is_numeric_dtype(df_cleaned[col])]

    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_cleaned[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns to generate a heatmap.")


    

    # -------------------------------
    # 3️⃣ Visual EDA
    # -------------------------------
    st.subheader("Visual EDA")

    # Load cleaned dataset from session state
    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    col1, col2 = st.columns([1, 2])

    # LEFT column: dropdowns
    with col1:
        plot_type = st.selectbox("Select Plot Type", [
            "Histogram", "Boxplot", "Countplot", "Scatter Plot", "Line Plot", "Pairplot", "Heatmap"
        ])

        # Count how many rows of inputs are being shown
        dropdown_count = 1
        column_for_plot = None

        if plot_type in ["Histogram", "Boxplot"]:
            column_for_plot = st.selectbox("Select column", df.select_dtypes(include=['float64', 'int64']).columns)
            dropdown_count += 1

        elif plot_type == "Countplot":
            column_for_plot = st.selectbox("Select column", df.select_dtypes(include=['object', 'category', 'string']).columns)
            dropdown_count += 1

        elif plot_type in ["Scatter Plot", "Line Plot"]:
            x_col = st.selectbox("X-axis", df.select_dtypes(include=['float64', 'int64']).columns, key="x_axis")
            y_col = st.selectbox("Y-axis", df.select_dtypes(include=['float64', 'int64']).columns, key="y_axis")
            dropdown_count += 2


    # RIGHT column: apply dynamic top margin using custom CSS
    with col2:
        padding_px = 10 * dropdown_count  # Adjust multiplier as needed (50px/dropdown = safe baseline)

        st.markdown(
            f"""<div style='margin-top: {padding_px}px;'>""",
            unsafe_allow_html=True
        )

        if plot_type != "Pairplot":
            fig, ax = plt.subplots(figsize=(6, 4))

            if plot_type == "Histogram":
                sns.histplot(df[column_for_plot], kde=True, ax=ax)
                ax.set_title(f"Histogram of {column_for_plot}")

            elif plot_type == "Boxplot":
                sns.boxplot(x=df[column_for_plot], ax=ax)
                ax.set_title(f"Boxplot of {column_for_plot}")

            elif plot_type == "Countplot":
                sns.countplot(x=df[column_for_plot], ax=ax)
                ax.set_title(f"Countplot of {column_for_plot}")
                plt.xticks(rotation=45)

            elif plot_type == "Scatter Plot":
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")

            elif plot_type == "Line Plot":
                sns.lineplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_title(f"Line Plot: {x_col} vs {y_col}")
                plt.xticks(rotation=45)

            elif plot_type == "Heatmap":
                sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Heatmap")

            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Generating pairplot...")
            pairplot_fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
            pairplot_fig.fig.set_size_inches(8, 6)
            st.pyplot(pairplot_fig.fig)




    
    # 4️⃣ Data Preprocessing
    
    st.subheader("Data Preprocessing")

    # Load cleaned dataset from session state
    df_cleaned = st.session_state.get("cleaned_df", df.copy())
    
    # Handling Missing Values
    st.markdown("#### Handle Missing Values")

    missing_option = st.selectbox(
        "Choose a missing value strategy:",
        ["No Action", "Drop rows with missing values", "Impute missing values"]
    )

    if missing_option == "Drop rows with missing values":
        df_cleaned.dropna(inplace=True)
        st.success("Rows with missing values dropped.")


    elif missing_option == "Impute missing values":
        impute_method = st.radio("Select imputation method:", ["Mean", "Median", "Mode"])

        try:
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().sum() > 0:

                    if is_numeric_dtype(df_cleaned[col]):
                        if impute_method == "Mean":
                            val = df_cleaned[col].mean()
                            if is_integer_dtype(df_cleaned[col]):
                                st.caption(f"Note: Rounded mean for `{col}` to fit Int64 column.")
                                val = int(round(val))
                            df_cleaned[col] = df_cleaned[col].fillna(val)
                            st.write(f"`{col}` imputed with **mean**: {val}")
                    
                        elif impute_method == "Median":
                            val = df_cleaned[col].median()
                            if is_integer_dtype(df_cleaned[col]):
                                st.caption(f"Note: Rounded median for `{col}` to fit Int64 column.")
                                val = int(round(val))
                            df_cleaned[col] = df_cleaned[col].fillna(val)
                            st.write(f"`{col}` imputed with **median**: {val}")
                    
                        elif impute_method == "Mode":
                            val = df_cleaned[col].mode()[0]
                            df_cleaned[col] = df_cleaned[col].fillna(val)
                            st.write(f"`{col}` imputed with **mode**: {val}")
                
                    else:
                        val = df_cleaned[col].mode()[0]
                        df_cleaned[col] = df_cleaned[col].fillna(val)
                        st.write(f"`{col}` (non-numeric) imputed with **mode**: {val}")

            st.success(f"Missing values imputed using {impute_method.lower()}.")

        except Exception as e:
            st.error(f"Imputation failed: {str(e)}")
    
    st.session_state.cleaned_df = df_cleaned.copy()


    
    # Re-check missing values in the cleaned dataset
    st.markdown("###### Missing Values After Handling")

    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    missing_after = df_cleaned.isnull().sum()
    missing_percent_after = (missing_after / len(df_cleaned)) * 100
    missing_df_after = pd.DataFrame({
        "Missing Values": missing_after,
        "Percent (%)": missing_percent_after
    })
    missing_df_after = missing_df_after[missing_df_after["Missing Values"] > 0]

    if not missing_df_after.empty:
        st.dataframe(missing_df_after.astype(str))
    else:
        st.success("No missing values remaining in the cleaned dataset!")
    
    st.session_state.cleaned_df = df_cleaned.copy()

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    st.markdown("#### Feature Scaling")

    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    scaling_option = st.selectbox(
        "Choose a scaling method:",
        ["No Action", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer"]  
    )

    try:
        if scaling_option != "No Action":
            scaler_map = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
                "MaxAbsScaler": MaxAbsScaler(),
                "Normalizer": Normalizer()
            }

            scaler = scaler_map[scaling_option]

            numeric_cols = [col for col in df_cleaned.columns if is_numeric_dtype(df_cleaned[col])]

            if numeric_cols:
                df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
                st.success(f"Applied {scaling_option} to numeric columns.")
                st.write("Scaled columns:", numeric_cols)

                # 👇 Top 10 preview after scaling
                st.markdown("#### Preview After Scaling")
                st.dataframe(df_cleaned.head())

            else:
                st.warning("No numeric columns found for scaling.")

    except Exception as e:
        st.error(f"Scaling failed: {str(e)}")
    
    st.session_state.cleaned_df = df_cleaned.copy()



    # -------------------------------
    # Categorical Encoding
    # -------------------------------
    st.markdown("#### Categorical Encoding")

    df_cleaned = st.session_state.get("cleaned_df", df.copy())

    # Step 1: Identify categorical columns
    cat_cols = [col for col in df_cleaned.columns if is_object_dtype(df_cleaned[col]) or isinstance(df_cleaned[col].dtype, pd.CategoricalDtype) or df_cleaned[col].dtype.name == "string"]

    if not cat_cols:
        st.info("No categorical columns detected.")
    else:
        selected_encoding = st.selectbox(
            "Select encoding method:",
            ["No Action", "Label Encoding", "Ordinal Encoding", "One-Hot Encoding"]
        )

        selected_columns = st.multiselect(
            "Select categorical columns to encode:",
            cat_cols
        )

        if selected_encoding == "Ordinal Encoding":
            st.markdown("*Optional: Enter custom order for each selected column (comma-separated). Leave blank for default alphabetical order.*")

        try:
            for col in selected_columns:
                if selected_encoding == "Label Encoding":
                    le = LabelEncoder()
                    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
                    st.write(f"`{col}` encoded with **Label Encoding**.")

                elif selected_encoding == "Ordinal Encoding":
                    custom_order = st.text_input(f"Custom order for `{col}` (comma-separated)", key=f"order_{col}")
                    if custom_order:
                        categories = [x.strip() for x in custom_order.split(",")]
                        oe = OrdinalEncoder(categories=[categories])
                        df_cleaned[col] = oe.fit_transform(df_cleaned[[col]].astype(str))
                        st.write(f"`{col}` encoded with **Ordinal Encoding (custom order)**: {categories}")
                    else:
                        sorted_categories = sorted(df_cleaned[col].dropna().unique())
                        oe = OrdinalEncoder(categories=[sorted_categories])
                        df_cleaned[col] = oe.fit_transform(df_cleaned[[col]].astype(str))
                        st.write(f"`{col}` encoded with **Ordinal Encoding (default)**: {sorted_categories}")

                elif selected_encoding == "One-Hot Encoding":
                    df_cleaned = pd.get_dummies(df_cleaned, columns=[col], drop_first=True)
                    st.write(f"`{col}` encoded with **One-Hot Encoding**.")

            if selected_encoding != "No Action" and selected_columns:
                st.success("Encoding completed.")

                # Preview the updated dataset
                st.markdown("#### Preview After Encoding")
                st.dataframe(df_cleaned.head(10))
            

        except Exception as e:
            st.error(f"Encoding failed: {str(e)}")

    st.session_state.cleaned_df = df_cleaned.copy()



