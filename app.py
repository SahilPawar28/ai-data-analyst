import streamlit as st
import pandas as pd
from data_loader import load_data, get_column_types
from analysis_engine import run_full_analysis
from ai_insights import generate_insights
from query_engine import execute_query
from chart_generator import generate_chart_from_prompt, export_chart_image

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous AI Data Analyst")
st.markdown("Upload a dataset and let AI analyze, visualize, and explain it for you.")

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.markdown("**Supported formats:** CSV, Excel")
    st.markdown("**Example columns:** region, product, revenue, order_date")

# ─── Main Logic ────────────────────────────────────────────
if uploaded_file is not None:

    # Step 1: Load data
    df, error = load_data(uploaded_file)

    if error:
        st.error(error)
        st.stop()

    st.success(f"✅ Dataset loaded: **{uploaded_file.name}** — {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Detect column types
    column_types = get_column_types(df)

    # Step 3: Run full analysis
    with st.spinner("🔍 Analyzing dataset..."):
        analysis = run_full_analysis(df, column_types)

    # ── Tab Layout ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview",
        "📈 Statistics",
        "📉 Charts",
        "🧠 AI Insights",
        "💬 Ask a Question"
    ])

    # ── Tab 1: Dataset Overview ─────────────────────────────
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", analysis["summary"]["rows"])
        col2.metric("Columns", analysis["summary"]["columns"])
        col3.metric("Missing Values", analysis["summary"]["total_missing"])
        col4.metric("Numeric Cols", len(column_types["numeric"]))

        st.markdown("#### Column Types Detected")
        st.write(f"**Numeric:** {', '.join(column_types['numeric']) or 'None'}")
        st.write(f"**Categorical:** {', '.join(column_types['categorical']) or 'None'}")
        st.write(f"**Date:** {', '.join(column_types['date']) or 'None'}")

    # ── Tab 2: Statistics ───────────────────────────────────
    with tab2:
        st.subheader("Descriptive Statistics")
        if analysis["statistics"]:
            stats_df = pd.DataFrame(analysis["statistics"]).T
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numeric columns found for statistics.")

        st.subheader("Categorical Distributions")
        if analysis["distribution"]:
            for col, data in analysis["distribution"].items():
                with st.expander(f"📌 {col}"):
                    dist_df = pd.DataFrame({
                        "Value": list(data["counts"].keys()),
                        "Count": list(data["counts"].values()),
                        "Percentage (%)": list(data["percentages"].values())
                    })
                    st.dataframe(dist_df, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    # ── Tab 3: Charts ───────────────────────────────────────
    with tab3:
        st.subheader("🎨 Create a Custom Chart")
        st.markdown("Describe any chart in plain English and the AI will build it.")

        chart_prompt = st.text_input(
            "Describe the chart you want:",
            placeholder="e.g. Show average life expectancy by continent as a bar chart",
            key="chart_prompt_input",
        )

        if st.button("✨ Generate Chart") and chart_prompt:
            with st.spinner("🎨 Building your chart..."):
                fig, generated_code, chart_error = generate_chart_from_prompt(chart_prompt, df)

            st.session_state["custom_chart_fig"] = fig
            st.session_state["custom_chart_code"] = generated_code
            st.session_state["custom_chart_error"] = chart_error
            st.session_state["custom_chart_png"] = None
            st.session_state["custom_chart_jpg"] = None

        if st.session_state.get("custom_chart_error"):
            st.error(st.session_state["custom_chart_error"])
            if st.session_state.get("custom_chart_code"):
                with st.expander("🔧 Generated Code"):
                    st.code(st.session_state["custom_chart_code"], language="python")

        elif st.session_state.get("custom_chart_fig") is not None:
            fig = st.session_state["custom_chart_fig"]
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔧 Generated Code"):
                st.code(st.session_state["custom_chart_code"], language="python")

            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                if st.session_state.get("custom_chart_png") is None:
                    try:
                        st.session_state["custom_chart_png"] = export_chart_image(fig, "png")
                    except Exception as e:
                        st.session_state["custom_chart_png"] = e
                png_data = st.session_state["custom_chart_png"]
                if isinstance(png_data, Exception):
                    st.caption(f"PNG export failed: {png_data}")
                else:
                    st.download_button(
                        "⬇️ Download PNG", data=png_data,
                        file_name="chart.png", mime="image/png",
                    )

            with dl_col2:
                if st.session_state.get("custom_chart_jpg") is None:
                    try:
                        st.session_state["custom_chart_jpg"] = export_chart_image(fig, "jpg")
                    except Exception as e:
                        st.session_state["custom_chart_jpg"] = e
                jpg_data = st.session_state["custom_chart_jpg"]
                if isinstance(jpg_data, Exception):
                    st.caption(f"JPG export failed: {jpg_data}")
                else:
                    st.download_button(
                        "⬇️ Download JPG", data=jpg_data,
                        file_name="chart.jpg", mime="image/jpeg",
                    )

    # ── Tab 4: AI Insights ──────────────────────────────────
    with tab4:
        st.subheader("🧠 AI-Generated Business Insights")
        if st.button("✨ Generate Insights"):
            with st.spinner("Asking Gemini to analyze your data..."):
                insights, err = generate_insights(
                    analysis["summary"],
                    analysis["statistics"],
                    analysis["distribution"]
                )
            if err:
                st.error(err)
            else:
                st.markdown(insights)
                with st.expander("🛠️ Debug: View Raw AI Text"):
                    st.text(insights)

    # ── Tab 5: Natural Language Q&A ─────────────────────────
    with tab5:
        st.subheader("💬 Ask a Question About Your Data")
        st.markdown("Examples: *Which region has highest sales? What is the average revenue? Show top 5 products.*")

        question = st.text_input("Your question:", placeholder="Which product sells the most?")

        if st.button("🔍 Ask") and question:
            with st.spinner("Thinking..."):
                answer, generated_code = execute_query(question, df)

            st.markdown("#### Answer:")
            
            if isinstance(answer, pd.DataFrame):
                st.dataframe(answer, use_container_width=True)
            elif isinstance(answer, pd.Series):
                st.dataframe(answer.to_frame(), use_container_width=True)
            elif isinstance(answer, str) and answer.startswith("❌"):
                st.error(answer)
            else:
                st.success(str(answer))

            if generated_code:
                with st.expander("🔧 Generated Pandas Code"):
                    st.code(generated_code, language="python")

else:
    st.info("👈 Upload a dataset from the sidebar to get started.")
