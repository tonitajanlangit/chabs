
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
@st.cache_data
def load_data():
    file_path = "netflix_titles.csv"  # Update with your file path
    df = pd.read_csv(file_path)
    df_movies = df[df['type'] == 'Movie']  # Filter only movies
    return df_movies

df = load_data()

# Sidebar - Country Filter
st.sidebar.title("Filters")
all_countries = df['country'].dropna().unique().tolist()
selected_country = st.sidebar.selectbox("Select a Country", ["All"] + all_countries)

# Filter data based on selection
if selected_country != "All":
    df_filtered = df[df['country'] == selected_country]
else:
    df_filtered = df

# Count movies per country
movies_per_country = df_filtered['country'].value_counts()

# Main Title
st.title("📽️ Netflix Movies Dashboard")

# Show Movie Count Bar Chart
st.subheader("Number of Movies Per Country")
fig, ax = plt.subplots(figsize=(10, 5))
movies_per_country.plot(kind='bar', ax=ax, color="skyblue")
ax.set_xlabel("Country")
ax.set_ylabel("Number of Movies")
st.pyplot(fig)

# Show Table
st.subheader("Movie Data")
st.dataframe(df_filtered[['title', 'country', 'release_year', 'duration', 'rating']])

# Footer
st.markdown("🔍 **Use the sidebar to filter movies by country!**")
