import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile  # Added for zip handling

APP_TITLE = "LU Researchers Explorer"

@st.cache_data
def load_profiles():
    # Changed to relative path
    path = "profiless.csv"
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load profiles CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def load_articles():
    folder = "researcher_corpora"
    zip_path = "researcher_corpora.zip"  # Added zip path
    
    # Extract zip if folder doesn't exist
    if not os.path.exists(folder):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.success("Extracted researcher corpora from zip file")
        else:
            st.error(f"Required data not found: {folder} folder and {zip_path} both missing")
            return pd.DataFrame()
    
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            st.warning(f"Error reading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def compute_researcher_similarity(articles_df):
    """Compute cosine similarity between researchers based on publication topics"""
    if articles_df.empty or 'Classified_Topic' not in articles_df.columns:
        return None
    
    # Create researcher-topic matrix
    topic_matrix = pd.crosstab(
        articles_df['Researcher Name'],
        articles_df['Classified_Topic']
    )
    
    # Compute cosine similarity
    similarity_matrix = pd.DataFrame(
        cosine_similarity(topic_matrix),
        index=topic_matrix.index,
        columns=topic_matrix.index
    )
    return similarity_matrix

def display_similar_researchers(researcher_name, similarity_matrix, profiles_df, articles_df, top_n=5):
    """Display similar researchers with their profiles"""
    if researcher_name not in similarity_matrix.index:
        st.warning(f"No topic data available for {researcher_name}")
        return
        
    scores = similarity_matrix[researcher_name].sort_values(ascending=False)
    # Remove self and get top matches
    top_scores = scores.drop(researcher_name).head(top_n)
    
    if top_scores.empty:
        st.info("No similar researchers found")
        return
    
    st.subheader(f"Researchers Similar to {researcher_name}")
    st.caption("Similarity is calculated based on overlapping publication topics")
    
    for name, score in top_scores.items():
        # Get profile info
        profile = profiles_df[profiles_df['Name'].str.contains(name, case=False)]
        if not profile.empty:
            profile_url = profile['Profile URL'].values[0]
            citations = profile['Total Citations'].values[0]
            h_index = profile['H-index (Total)'].values[0]
            papers = profile['Total Papers'].values[0]
        else:
            profile_url = "#"
            citations = "N/A"
            h_index = "N/A"
            papers = "N/A"
        
        # Get researcher's topics
        researcher_topics = articles_df[articles_df['Researcher Name'] == name]
        topics = researcher_topics['Classified_Topic'].value_counts().head(3)
        top_topics = ", ".join(topics.index.tolist()) if not topics.empty else "No topics available"
        
        # Create profile link
        if profile_url and profile_url != "#":
            name_display = f"[{name}]({profile_url})"
        else:
            name_display = name
        
        with st.container():
            st.markdown(f"#### {name_display}")
            st.markdown(f"**Similarity:** `{score:.3f}`")
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("H-Index", h_index)
            with cols[1]:
                st.metric("Citations", citations)
            with cols[2]:
                st.metric("Papers", papers)
            with cols[3]:
                st.metric("Top Topics", top_topics)
                
            st.progress(float(score))
            st.markdown("---")

def display_profiles(profiles_df, author_query):
    matching_profiles = profiles_df[profiles_df['Name'].str.lower().str.contains(author_query)]
    if not matching_profiles.empty:
        for _, row in matching_profiles.iterrows():
            name = row['Name']
            profile_url = row.get('Profile URL', '#')
            citations = row.get('Total Citations', 'N/A')
            papers = row.get('Total Papers', 'N/A')
            h_index = row.get('H-index (Total)', 'N/A')

            if profile_url and profile_url != '#':
                name_link = f'<a href="{profile_url}" target="_blank" class="author-link">{name}</a>'
            else:
                name_link = name

            st.markdown(f"### {name_link}", unsafe_allow_html=True)

            st.markdown(
                f"<div style='display:flex; gap: 20px; flex-wrap: wrap;'>"
                f"<span><b>Total Citations:</b> {citations}</span>"
                f"<span><b>Total Papers:</b> {papers}</span>"
                f"<span><b>H-Index:</b> {h_index}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No matching author profile found.")

def display_articles(articles_df):
    for author in articles_df['Researcher Name'].unique():
        author_articles = articles_df[articles_df['Researcher Name'] == author]
        for _, row in author_articles.iterrows():
            title = row.get('Title', 'No Title')
            topic = row.get('Classified_Topic', 'No Classification')
            link = row.get('Scholar Link', '#')
            description = row.get('Description', 'No Description')
            co_authors = row.get('Authors', 'N/A')
            citations = row.get('Total citations', 'N/A')

            with st.expander(title, expanded=False):
                st.markdown(f"**Article Link:** <a href='{link}' target='_blank' style='color:#276678;'>click here</a>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Classified Topic:** <code style='background:#7da9c0; color:#073b4c;'>{topic}</code>", unsafe_allow_html=True)
                st.markdown(f"**Co-authors:** {co_authors}")
                st.markdown(f"**Cited by:** {citations}")

def publication_forecast(articles_df, forecast_years=5):
    # Check if publication date column exists
    if 'Publication date' not in articles_df.columns:
        st.warning("Publication date information not available for forecasting")
        return None
    
    # Extract year from publication date
    try:
        articles_df['Year'] = pd.to_datetime(articles_df['Publication date'], errors='coerce').dt.year
        articles_df = articles_df.dropna(subset=['Year'])
        articles_df['Year'] = articles_df['Year'].astype(int)
    except Exception as e:
        st.error(f"Error processing publication dates: {e}")
        return None
    
    # Count publications per year
    pub_counts = articles_df['Year'].value_counts().sort_index()
    
    # Need at least 3 years of data for forecasting
    if len(pub_counts) < 3:
        st.warning("Insufficient data for forecasting (need at least 3 years of publications)")
        return None
    
    # Prepare data for forecasting
    years = pub_counts.index.values.reshape(-1, 1)
    counts = pub_counts.values
    
    # Create polynomial features for better trend fitting
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(years)
    
    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, counts)
    
    # Forecast next N years
    last_year = years.max()
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    forecast = model.predict(future_years_poly)
    
    # Ensure no negative forecasts
    forecast = np.where(forecast < 0, 0, forecast)
    
    # Create results DataFrame
    history_df = pd.DataFrame({
        'Year': years.flatten(),
        'Publications': counts,
        'Type': 'Historical'
    })
    
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Publications': forecast,
        'Type': 'Forecast'
    })
    
    return pd.concat([history_df, forecast_df])

def set_background():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #12283a;
    }

    .author-link {
        color: #1b3559;
        text-decoration: none;      /* no underline */
        font-weight: 700;
        transition: color 0.3s ease, text-decoration 0.3s ease;
    }
    .author-link:hover {
        color: #0a2540;
        text-decoration: underline; /* underline on hover */
    }

    code {
        background-color: #7da9c0;
        padding: 3px 7px;
        border-radius: 5px;
        color: #073b4c;
        font-weight: 600;
    }

    .stSidebar {
        background-color: #d4e2f4;
        color: #12283a;
        font-weight: 600;
    }

    a {
        color: #1b3559;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    a:hover {
        color: #0a2540;
        text-decoration: underline;
    }

    .streamlit-expanderHeader {
        font-weight: 700;
        color: #1b3559;
    }

    .stInfo, .stWarning, .stError {
        font-weight: 600;
    }
    
    /* Forecast chart styling */
    .forecast-chart {
        margin-top: 30px;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Similar researcher cards */
    .similar-researcher-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .similar-researcher-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #1b3559;
    }
    
    /* Similarity search form */
    .similarity-form {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    set_background()
    st.title(APP_TITLE)

    profiles_df = load_profiles()
    articles_df = load_articles()
    
    # Compute researcher similarity matrix
    with st.spinner("Analyzing researcher similarities..."):
        similarity_matrix = compute_researcher_similarity(articles_df)

    st.sidebar.header("Search & Filter")
    author_input = st.sidebar.text_input("Author Name (for Profile & Articles)")
    classifications = sorted(articles_df['Classified_Topic'].dropna().unique())
    selected_classification = st.sidebar.selectbox("Filter by Classification", options=["All"] + classifications)
    
    # Similar researcher section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Researcher Similarity")
    if similarity_matrix is not None:
        researcher_list = sorted(similarity_matrix.index)
        similarity_researcher = st.sidebar.selectbox(
            "Find researchers similar to:",
            options=["Select a researcher"] + researcher_list
        )
        top_n = st.sidebar.slider("Number of similar researchers", 1, 10, 5)
    else:
        st.sidebar.info("Topic data not available for similarity analysis")
        similarity_researcher = ""

    # Forecasting options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Forecasting Options")
    forecast_years = st.sidebar.slider("Years to forecast", 3, 10, 5)
    show_forecast = st.sidebar.checkbox("Show publication forecast", True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This application allows you to explore researchers from the Lebanese University. "
        "You can search for an author to view their profile and published articles, "
        "filter articles by topic, find similar researchers, and see publication forecasts."
    )

    filtered_articles = articles_df.copy()

    # Researcher similarity standalone feature
    if similarity_researcher and similarity_researcher != "Select a researcher":
        st.markdown("""<h2 class='section-header'>Researcher Similarity Analysis</h2>""", unsafe_allow_html=True)
        
        # Display the selected researcher's profile
        st.subheader(f"Selected Researcher: {similarity_researcher}")
        display_profiles(profiles_df, similarity_researcher.lower())
        
        # Display similar researchers
        display_similar_researchers(
            similarity_researcher,
            similarity_matrix,
            profiles_df,
            articles_df,
            top_n
        )
        st.markdown("---")
        return  # Stop execution here to show only similarity results

    # Regular author profile and articles section
    if author_input:
        author_query = author_input.strip().lower()
        if ' ' not in author_query:
            filtered_articles['First_Name'] = filtered_articles['Researcher Name'].apply(
                lambda x: str(x).split()[0].lower() if pd.notna(x) else ''
            )
            filtered_articles = filtered_articles[filtered_articles['First_Name'] == author_query]
        else:
            filtered_articles = filtered_articles[
                filtered_articles['Researcher Name'].str.lower().str.contains(author_query)
            ]
    else:
        author_query = ""

    if selected_classification != "All":
        filtered_articles = filtered_articles[
            filtered_articles['Classified_Topic'] == selected_classification
        ]

    if author_query:
        st.markdown("""<h2 class='section-header'>Author Profile</h2>""", unsafe_allow_html=True)
        display_profiles(profiles_df, author_query)

    if author_query or selected_classification != "All":
        st.markdown("""<h2 class='section-header'>Articles</h2>""", unsafe_allow_html=True)
        if filtered_articles.empty:
            st.info("No articles found with current filters.")
        else:
            display_articles(filtered_articles)
            
            # Show forecast if enabled
            if show_forecast:
                st.markdown("""<h2 class='section-header'>Publication Forecast</h2>""", unsafe_allow_html=True)
                forecast_data = publication_forecast(filtered_articles, forecast_years)
                
                if forecast_data is not None:
                    # Display forecast chart
                    st.markdown("<div class='forecast-chart'>", unsafe_allow_html=True)
                    
                    # Create chart with y-axis starting at 0
                    chart_data = forecast_data.set_index('Year')['Publications']
                    st.line_chart(
                        chart_data, 
                        use_container_width=True,
                        color="#073b4c"
                    )
                    
                    # Add forecast explanation
                    st.caption("""
                    **Forecast Methodology:**
                    - Based on historical publication patterns
                    - Uses polynomial regression to capture trends
                    - Predicts future publication counts based on past data
                    - Ensures no negative publication counts
                    """)
                    
                    # Show forecast table with non-negative values
                    with st.expander("View forecast details"):
                        forecast_table = forecast_data.copy()
                        forecast_table['Publications'] = forecast_table['Publications'].round().astype(int)
                        st.dataframe(forecast_table)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please use the sidebar controls to explore researchers and articles.")

if __name__ == "__main__":
    main()
