from datetime import date
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from ipyleaflet import Map, Marker
from ipywidgets import HTML
import folium
from streamlit_folium import st_folium  
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import plotly.graph_objects as go
import re
import nltk
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string 
import googlemaps
import time
from folium.plugins import MarkerCluster
import itertools

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ----------------- STREAMLIT PAGE CONFIG & CUSTOM STYLES ----------------- #
st.set_page_config(
    page_title="PopIn Data Analysis Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Altair color scheme
PRIMARY_COLOR = "#fc6c64"  # Coral Red
WHITE = "#ffffff"  # White
COLORS = ["#3aafa9", "#fc6c64", "#f4d03f"]  # Chart colors
PASTEL = ["#ff9999", "#dda0dd", "#20b2aa"]  # For variation

# Load dataset
df_popin = pd.read_csv('MeetUp_PopIn_Events.csv')
df_popin.columns = df_popin.columns.str.replace(' ', '_')

def custom_theme():
    return {
        "config": {
            "view": {"height": 400, "width": 700},
            "mark": {"color": PRIMARY_COLOR},
            "axis": {"domainColor": WHITE, "gridColor": "#e0e0e0"},
        }
    }
alt.themes.register('custom_theme', custom_theme)
alt.themes.enable('custom_theme')

def apply_custom_styles():
    st.markdown(
        f"""
        <style>
            /* Main content background remains white */
            .main {{ background-color: white; }}
            /* Sidebar background uses the primary color */
            [data-testid="stSidebar"] {{ background-color: {PRIMARY_COLOR}; }}
            h1, h2, h3, h4, h5, h6, p, label {{ color: black !important; }}
            div[data-testid="stMetric"] {{
                background-color: {PRIMARY_COLOR}; 
                padding: 10px; 
                border-radius: 10px; 
                color: white; 
                text-align: center;
            }}
            /* Basic styling for any remaining buttons */
            .stButton > button {{
                width: 100%;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
                background-color: white !important;
                color: black !important;
                cursor: pointer;
                border: 1px solid black;
            }}
            .stButton > button:hover {{
                background-color: #f0f0f0 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# ----------------- SIDEBAR NAVIGATION WITH PERSISTENT HIGHLIGHTING ----------------- #
with st.sidebar:
    st.title('PopIn Data Analysis Filters')
    event_category = st.selectbox("Select Category", ["All", "Business", "Entertainment", "Other"])
    
    st.subheader("Select Visualization")
    event_buttons = [
        "Event Performance Overview",
        "Category Analysis",
        "Event Popularity",
        "Event Trends Over Time",
        "Host Analysis",
        "Event Location Insights",
        "Word Cloud"
    ]

    # Initialize session state if not already set
    if 'graph_selection' not in st.session_state:
        st.session_state.graph_selection = event_buttons[0]

    # Define a function to render each navigation button.
    def render_nav_button(label):
        if label == st.session_state.graph_selection:
            # Render a highlighted non-clickable div if this is the current selection.
            st.markdown(
                f"""
                <div style="
                    width: 100%;
                    background-color: #ffcccb;
                    color: black;
                    text-align: center;
                    padding: 10px;
                    border: 2px solid #ff5733;
                    border-radius: 5px;
                    margin: 5px 0;
                ">{label}</div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Render as a clickable button.
            if st.button(label, key=label):
                st.session_state.graph_selection = label

    # Loop through and render each navigation option.
    for button in event_buttons:
        render_nav_button(button)

# Optionally, display the currently selected view in the main area.
# st.subheader(f"Selected View: {st.session_state.graph_selection}")

#--------DATA PROCESSING HERE-------#

# Filter dataset based on category selection
if event_category != "All":
    df_popin = df_popin[df_popin['Category'] == event_category]

if 'Date_&_Time' in df_popin.columns:
    df_popin['Date'] = df_popin['Date_&_Time'].str.split('T').str[0]
    df_popin['Time'] = df_popin['Date_&_Time'].str.split('T').str[1]
    #df_popin.drop(columns=['Date_&_Time'], inplace=True)              # @Mea, I commented this part because all my codes are dependent to index Date_&_Time

df_popin['Date'] = pd.to_datetime(df_popin['Date'], errors='coerce')
df_popin['Month-Year'] = df_popin['Date'].dt.to_period('M')

#--------------Mica Data Processing-----------------------------------------------------------------------------------#

#-------1 Will be used for the Day of the Week 1----------#

# Convert 'Date & Time' to datetime format with timezone awareness
df_popin['Date_&_Time1'] = pd.to_datetime(df_popin['Date_&_Time'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')
df_popin['Date_&_Time2'] = pd.to_datetime(df_popin['Date_&_Time1'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')
# Check if conversion was successful
print(df_popin.dtypes)  # This should show 'Date & Time' as datetime64[ns, UTC] or datetime64[ns, tz]

# Extract date and time separately
df_popin['Event Date'] = df_popin['Date_&_Time2'].dt.date.astype(str) 
df_popin['Event Time'] = df_popin['Date_&_Time2'].dt.time.astype(str)
df_popin['Event Day'] = df_popin['Date_&_Time2'].dt.day_name()

#---------------------1 end 1--------------------------------#

#-------2 Will be used for the Location Insights 2----------#
### --- CLEANING LOCATION COLUMN --- ###

# Function to remove punctuation except commas
def remove_punctuation_except_comma(text):
    if isinstance(text, str):
        no_punct = [char if char == ',' else char for char in text if char not in string.punctuation or char == ',']
        return ''.join(no_punct)
    return text

df_popin['Location'] = df_popin['Location'].apply(remove_punctuation_except_comma)

# Function to clean location formatting (replace middle dots, remove extra spaces)
def clean_location(text):
    if isinstance(text, str):
        text = text.replace('·', ',')  # Replace middle dot with comma
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()
    return text

df_popin['Location'] = df_popin['Location'].apply(clean_location)

# Function to standardize virtual/online locations
def replace_virtual_online(Location):
    if isinstance(Location, str):
        if re.search(r'\b(virtual|online)\b', Location, re.IGNORECASE):
            return "Online"
    return Location

df_popin['Location'] = df_popin['Location'].apply(replace_virtual_online)

### ---  GEOCODING (Convert Location to Latitude & Longitude) --- ###

# Initialize Google Maps API (Replace with a real API key)
API_KEY = "AIzaSyCM9eOCLJbry0h1RuUIcze2j2lVIXbRoi4"  # 🔴 Replace this with your actual API key!
gmaps = googlemaps.Client(key=API_KEY)

# Function to get latitude and longitude from an address
def get_lat_lon(Location, max_retries=3):
    if isinstance(Location, str) and Location.lower() == "online":  # Skip "online" locations
        return pd.Series([None, None])

    for attempt in range(max_retries):
        try:
            geocode_result = gmaps.geocode(Location)
            if geocode_result:
                lat = geocode_result[0]['geometry']['location']['lat']
                lon = geocode_result[0]['geometry']['location']['lng']
                return pd.Series([lat, lon])
        except Exception as e:
            print(f"Error for {Location}: {e}")
            time.sleep(2)  # Wait before retrying
            continue

    return pd.Series([None, None])  # Return None if all retries fail

# Apply geocoding with a delay
if "latitude" not in df_popin.columns or "longitude" not in df_popin.columns:
    df_popin[["latitude", "longitude"]] = df_popin["Location"].apply(lambda x: get_lat_lon(x))


# Ensure latitude and longitude are numeric and drop rows where they are missing
df_popin["latitude"] = pd.to_numeric(df_popin["latitude"], errors="coerce")
df_popin["longitude"] = pd.to_numeric(df_popin["longitude"], errors="coerce")

# Drop rows where latitude or longitude is missing
#df_popin = df_popin.dropna(subset=["latitude", "longitude"]) # Change to comment, cannot drop not lat or ltn because of online loc

#----------------------2 End 2---------------------#

#-------------------------------------------- End Mica Data Processing-------------------------------------------------#

#---------------------------------- For Marla's ----------------------------------------------------------------------------#
#-----------------FOR RECHECKING-------------------#
# Add 'Event Type' (Online vs In-Person)
df_popin["Event_Type"] = df_popin["Location"].apply(lambda x: "Online" if "online" in str(x).lower() else "In-Person")   

# Function to categorize events based on keywords
def categorize_event(name):
    name = str(name).lower()
    if any(word in name for word in ["tech", "ai", "data", "python", "coding", "programming"]):
        return "Technology"
    elif any(word in name for word in ["business", "entrepreneur", "startup", "marketing", "finance"]):
        return "Business & Networking"
    elif any(word in name for word in ["health", "wellness", "yoga", "mental", "meditation"]):
        return "Health & Wellness"
    elif any(word in name for word in ["meetup", "hangout", "social", "party", "friend", "fun"]):
        return "Social"
    elif any(word in name for word in ["career", "job", "resume", "interview", "hiring", "networking"]):
        return "Career & Jobs"
    elif any(word in name for word in ["music", "concert", "dj", "band"]):
        return "Music & Entertainment"
    elif any(word in name for word in ["fitness", "hiking", "cycling", "outdoor"]):
        return "Sports & Outdoors"
    elif any(word in name for word in ["education", "seminar", "workshop", "learning"]):
        return "Education & Learning"
    else:
        return "Other"

# Apply event categorization
df_popin["Event_Category"] = df_popin["Event_Name"].apply(categorize_event)

#-----------------END FOR RECHECKING-------------------#

#--------------------------------End of Marla's-----------------------------------------------------------#

# ----------------- DISPLAY VISUALIZATIONS BASED ON BUTTON SELECTION ----------------- #
# ----------------- EVENT PERFORMANCE OVERVIEW ----------------- #
if st.session_state.graph_selection == "Event Performance Overview":
    #Visual 1
    st.subheader("📊 Event Performance Overview")
    total_events = len(df_popin)
    total_attendees = df_popin['Attendees'].sum() if 'Attendees' in df_popin.columns else 0
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🏆 Total Events", value=total_events)
    with col2:
        st.metric(label="👥 Total Attendees", value=total_attendees)
   
   #Visual 2
    st.subheader("📈 Average Attendance Per Event Category") 
    if not df_popin.empty:
        average_attendance = df_popin.groupby('Category')['Attendees'].mean().reset_index()
        average_attendance.columns = ['Category', 'Average Attendance']
        avg_attendance_chart = alt.Chart(average_attendance).mark_bar().encode(
            x=alt.X("Average Attendance:Q", title="Average Attendance"),
            y=alt.Y("Category:N", sort='-x', title="Category"),
            tooltip=["Category", "Average Attendance"],
            color=alt.Color("Category:N", scale=alt.Scale(domain=average_attendance['Category'].tolist(), range=COLORS)),
            text=alt.Text("Average Attendance:Q", format=".0f")
        ).properties(width=750, height=400)
        avg_annotations = alt.Chart(average_attendance).mark_text(
            align='left',
            dx=10, dy=-10,
            fontSize=12,
            fontWeight='bold',
            color='black'
        ).encode(
            x=alt.X('Average Attendance:Q'),
            y=alt.Y('Category:N', sort='-x'),
            text=alt.Text('Average Attendance:Q', format='.0f')
        )
        avg_final_chart = (avg_attendance_chart + avg_annotations).configure_mark(opacity=0.8)
        st.altair_chart(avg_final_chart, use_container_width=True)
    else:
        st.warning("No data available for the selected category.")
    
    #Visual 3
    # --- Donut Chart for Online vs. In-Person --- #
    df_popin["Location Type"] = df_popin["Location"].apply(lambda x: "Online" if x.lower() == "online" else "In-Person")
    location_counts = df_popin["Location Type"].value_counts().reset_index()
    location_counts.columns = ["Location Type", "Count"]
    location_counts["Percentage"] = (location_counts["Count"] / location_counts["Count"].sum()) * 100
    def make_donut_chart(input_df, category_col, value_col, percentage_col, title):
        chart = alt.Chart(input_df).mark_arc(innerRadius=50, cornerRadius=15).encode(
            theta=alt.Theta(f"{value_col}:Q"),
            color=alt.Color(f"{category_col}:N", scale=alt.Scale(scheme="category10")),
            tooltip=[category_col, value_col]
        )
        text = alt.Chart(input_df).mark_text(size=14, fontWeight="bold", color="white").encode(
            theta=alt.Theta(f"{value_col}:Q"),
            text=alt.Text(f"{percentage_col}:Q", format=".1f"),
            color=alt.Color(f"{category_col}:N", scale=alt.Scale(scheme="category10")),
            radius=alt.value(80)
        )
        return (chart + text).properties(width=300, height=300, title=title)
    
    st.subheader("📍 Online vs. In-Person Events")
    st.altair_chart(make_donut_chart(location_counts, "Location Type", "Count", "Percentage", "Event Distribution"), use_container_width=True)

# ----------------- CATEGORY ANALYSIS ----------------- #
elif st.session_state.graph_selection == "Category Analysis":
    st.subheader("📊 Category Analysis")

    # ---  Event Count by Category ---
    if not df_popin.empty:
        st.subheader("🎭 Event Count by Category")
        event_count = df_popin['Category'].value_counts().reset_index()
        event_count.columns = ['Category', 'Count']

        event_count_chart = alt.Chart(event_count[:20]).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of Events"),
            y=alt.Y("Category:N", sort='-x', title="Category"),
            tooltip=["Category", "Count"],
            color=alt.Color("Category:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=750, height=400)

        st.altair_chart(event_count_chart, use_container_width=True)

    # ---  Total Attendees by Category ---
        st.subheader("📈 Total Attendees by Category")
        total_attendance = df_popin.groupby('Category')['Attendees'].sum().reset_index()
        total_attendance.columns = ['Category', 'Total Attendees']

        total_attendance_chart = alt.Chart(total_attendance).mark_bar(cornerRadius=4).encode(
            y=alt.Y("Category:N", title="Category", sort='-x'),
            x=alt.X("Total Attendees:Q", title="Total Attendees"),
            tooltip=["Category", "Total Attendees"],
            color=alt.Color("Category:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=750, height=400)

        st.altair_chart(total_attendance_chart, use_container_width=True)

    #--------------------------MARLA-----------------------------------#
    #-----------------FOR RECHECKING-------------------# comment: IT SHOULD NOT BE ONLINE VS IN PERSON, SHOULD BE ENTERTAINMENT VS BUSINESS VS OTHERS
    # ---  Category Distribution Overall ---   # Generated from Marla
    st.subheader("📊 Event Category Distribution")
    category_counts = df_popin["Event_Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    category_chart = alt.Chart(category_counts).mark_bar().encode(
        x=alt.X("Count:Q", title="Number of Events"),
        y=alt.Y("Category:N", sort='-x', title="Event Category"),
        color=alt.Color("Category:N", scale=alt.Scale(scheme='category20b'))
    ).properties(width=750, height=400)

    st.altair_chart(category_chart, use_container_width=True)

    # ---  Online Events - Category Distribution ---
    st.subheader("📊 Category Distribution for Online Events")
    df_online = df_popin[df_popin["Event_Type"] == "Online"]
    
    if df_online.empty:
        st.warning("No online events found.")
    else:
        category_counts_online = df_online["Event_Category"].value_counts().reset_index()
        category_counts_online.columns = ["Category", "Count"]

        online_chart = alt.Chart(category_counts_online).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of Online Events"),
            y=alt.Y("Category:N", sort='-x', title="Event Category"),
            color=alt.Color("Category:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=750, height=400)

        st.altair_chart(online_chart, use_container_width=True)

    # ---  In-Person Events - Category Distribution ---
    st.subheader("📊 Category Distribution for In-Person Events")
    df_in_person = df_popin[df_popin["Event_Type"] == "In-Person"]
    
    if df_in_person.empty:
        st.warning("No in-person events found.")
    else:
        category_counts_in_person = df_in_person["Event_Category"].value_counts().reset_index()
        category_counts_in_person.columns = ["Category", "Count"]

        in_person_chart = alt.Chart(category_counts_in_person).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of In-Person Events"),
            y=alt.Y("Category:N", sort='-x', title="Event Category"),
            color=alt.Color("Category:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=750, height=400)

        st.altair_chart(in_person_chart, use_container_width=True)
        
#-----------------END FOR RECHECKING-------------------#

# ----------------- EVENT POPULARITY ----------------- #
elif st.session_state.graph_selection == "Event Popularity":
    st.subheader("⭐ Event Popularity Insights")

    #defining Event_Type NOTE: this is also used for wordcloud    
    df_popin['Event_Type'] = df_popin['Location'].apply(lambda x: 'Online' if 'Online' in str(x) else 'In-Person')    
    top_in_person = df_popin[df_popin['Event_Type'] == 'In-Person'].nlargest(10, 'Attendees')
    top_online = df_popin[df_popin['Event_Type'] == 'Online'].nlargest(10, 'Attendees')
    least_in_person = df_popin[df_popin['Event_Type'] == 'In-Person'].nsmallest(10, 'Attendees')
    least_online = df_popin[df_popin['Event_Type'] == 'Online'].nsmallest(10, 'Attendees')

    def create_bar_chart(data, title):
        bars = alt.Chart(data).mark_bar(color="#fcd7d3").encode(
            x=alt.X("Attendees:Q", title="Number of Attendees"),
            y=alt.Y("Event_Name:N", sort='-x', title="Event Name"),
            tooltip=["Event_Name", "Attendees"]
        ).interactive()
        text = bars.mark_text(
            align='left', baseline='middle', dx=3, color="black"
        ).encode(text="Attendees:Q")
        return (bars + text).properties(title=title, width=350, height=400)

    def create_vertical_bar_chart(data, title):
        bars = alt.Chart(data).mark_bar(color="#a6cee3").encode(
            x=alt.X("Event_Name:N",
                    sort=alt.EncodingSortField(field="Attendees", op="sum", order="descending"),
                    title="Event Name"),
            y=alt.Y("Attendees:Q", title="Number of Attendees"),
            tooltip=["Event_Name", "Attendees"]
        ).interactive()
        text = bars.mark_text(
            align='center', baseline='bottom', dy=-5, color="black"
        ).encode(text="Attendees:Q")
        return (bars + text).properties(title=title, width=350, height=400)

    col3, col4 = st.columns(2)
    with col3:
        #Visual 6
        st.altair_chart(create_bar_chart(top_in_person, "Top 10 Most Attended In-Person Events"), use_container_width=True)
    with col4:
        #Visual 7
        st.altair_chart(create_bar_chart(top_online, "Top 10 Most Attended Online Events"), use_container_width=True)
    col5, col6 = st.columns(2)
    with col5:
        #Visual 8
        st.altair_chart(create_vertical_bar_chart(least_in_person, "Top 10 Least Attended In-Person Events"), use_container_width=True)
    with col6:
        #Visual 9
        st.altair_chart(create_vertical_bar_chart(least_online, "Top 10 Least Attended Online Events"), use_container_width=True)

# ----------------- EVENT TRENDS OVER TIME ----------------- #
elif st.session_state.graph_selection == "Event Trends Over Time":
    st.subheader("📈 Event Trends Over Time")

    # Work with a copy to avoid affecting other sections
    df_events_trends = df_popin.copy()

    df_events_trends["Date"] = pd.to_datetime(df_events_trends["Date"], errors='coerce').dt.date
    events_over_time = df_events_trends.groupby("Date").size().reset_index(name="Event_Count")

    line_events = alt.Chart(events_over_time).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Event_Count:Q", title="Number of Events"),
        tooltip=["Date", "Event_Count"]
    ).properties(title="Event Trends Over Time", width=700, height=400)

    # Visual 10
    st.altair_chart(line_events, use_container_width=True)

    #-----MICA'S EVENTS DAYS OF THE WEEK-------#
    # Days of the Week

    # Group data by Event Day & Category
    attendees_by_day_category = df_popin.groupby(["Event Day", "Category"])["Attendees"].sum().reset_index()

    # Ensure Days are sorted properly (not alphabetically)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    attendees_by_day_category["Event Day"] = pd.Categorical(attendees_by_day_category["Event Day"], categories=day_order, ordered=True)

    # Sort Data by Event Day
    attendees_by_day_category = attendees_by_day_category.sort_values("Event Day")

    # STREAMLIT DASHBOARD
    st.title("📊 Attendees: Day of the Week vs Categories")

    # Bar Chart 
    st.subheader("📊 Bar Chart: Attendees by Day and Category")
    bar_chart = alt.Chart(attendees_by_day_category).mark_bar().encode(
        x=alt.X("Event Day:N", title="Day of the Week", sort=day_order),
        y=alt.Y("Attendees:Q", title="Total Attendees"),
        color=alt.Color("Category:N", title="Category"),
        tooltip=["Event Day", "Category", "Attendees"]
    ).interactive()

    st.altair_chart(bar_chart, use_container_width=True)

    # Events: days vs category

    # Group data by Event Day & Category 
    events_by_day_category = df_popin.groupby(["Event Day", "Category"]).size().reset_index(name="Event Count")

    # Ensure Days are sorted properly 
    events_by_day_category["Event Day"] = pd.Categorical(events_by_day_category["Event Day"], categories=day_order, ordered=True)

    # Sort Data by Event Day
    events_by_day_category = events_by_day_category.sort_values("Event Day")

    # STREAMLIT DASHBOARD
    st.title("📊 Number of Events: Day of the Week vs Categories")

    # Line Chart for Number of Events
    st.subheader("📈 Number of Events Over Days of the Week by Category")
    events_line_chart = alt.Chart(events_by_day_category).mark_line(point=True).encode(
        x=alt.X("Event Day:N", title="Day of the Week", sort=day_order),
        y=alt.Y("Event Count:Q", title="Number of Events"),
        color=alt.Color("Category:N", title="Category"),
        tooltip=["Event Day", "Category", "Event Count"]
    ).interactive()

    st.altair_chart(events_line_chart, use_container_width=True)

    # Best Day Or Worst Day

    # Ensure all combinations of Event Day & Category exist (fills missing days)
    all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    all_categories = df_popin["Category"].unique()
    full_reference = pd.DataFrame(itertools.product(all_days, all_categories), columns=["Event Day", "Category"])

    # Aggregate event count and attendees
    attendees_by_day_category = df_popin.groupby(["Event Day", "Category"])["Attendees"].sum().reset_index()
    events_by_day_category = df_popin.groupby(["Event Day", "Category"]).size().reset_index(name="Event Count")

    # Merge data into a full reference table 
    data_by_day_category = pd.merge(full_reference, attendees_by_day_category, on=["Event Day", "Category"], how="left").fillna(0)
    data_by_day_category = pd.merge(data_by_day_category, events_by_day_category, on=["Event Day", "Category"], how="left").fillna(0)

    # Function to find all best/worst days
    def find_best_worst_days(df, value_column, group_column):
        """Finds all best and worst days per category."""
        best_days = df[df.groupby(group_column)[value_column].transform(max) == df[value_column]]
        worst_days = df[df.groupby(group_column)[value_column].transform(min) == df[value_column]]
        return best_days, worst_days

    # Find Best & Worst Days for Attendees
    best_attendance, worst_attendance = find_best_worst_days(data_by_day_category, "Attendees", "Category")

    # Find Best & Worst Days for Events
    best_events, worst_events = find_best_worst_days(data_by_day_category, "Event Count", "Category")

    # Merge data to create final tables
    best_day_table = pd.merge(best_attendance, best_events, on="Category", suffixes=("_Attendees", "_Events"))
    worst_day_table = pd.merge(worst_attendance, worst_events, on="Category", suffixes=("_Attendees", "_Events"))

    # Remove duplicate days by converting lists to unique sets & sorting them
    def unique_sorted_days(day_list):
        return ", ".join(sorted(set(day_list), key=lambda x: all_days.index(x)))  # Sort by week order

    # Apply the function to clean up duplicate days
    best_day_table = best_day_table.groupby("Category", as_index=False).agg({
        "Event Day_Attendees": lambda x: unique_sorted_days(x),
        "Attendees_Attendees": "first",  # First value since it's the same across grouped rows
        "Event Day_Events": lambda x: unique_sorted_days(x),
        "Event Count_Events": "first"
    })

    worst_day_table = worst_day_table.groupby("Category", as_index=False).agg({
        "Event Day_Attendees": lambda x: unique_sorted_days(x),
        "Attendees_Attendees": "first",
        "Event Day_Events": lambda x: unique_sorted_days(x),
        "Event Count_Events": "first"
    })

    # STREAMLIT DASHBOARD
    st.subheader("🏆 Best Day(s) for Each Category (Attendance & Events)")
    st.dataframe(best_day_table.rename(columns={
        "Event Day_Attendees": "Best Day (Attendees)", "Attendees_Attendees": "Number of Attendees",
        "Event Day_Events": "Best Day (Events)", "Event Count_Events": "Number of Events"
    }))

    st.subheader("📉 Worst Day(s) for Each Category (Attendance & Events)")
    st.dataframe(worst_day_table.rename(columns={
        "Event Day_Attendees": "Worst Day (Attendees)", "Attendees_Attendees": "Number of Attendees",
        "Event Day_Events": "Worst Day (Events)", "Event Count_Events": "Number of Events"
    }))
    #------------END OF MIC'S CODE FOR DAY OF THE WEEK---------


# ----------------- START FOR UPDATED HOST ANALYSIS ----------------- #

# HOST ANALYSIS HERE
# ----------------- HOST ANALYSIS ----------------- #
if st.session_state.graph_selection == "Host Analysis":
    st.subheader("👤 Host Analysis")
    st.subheader("📊 Top Event Hosts")

    # Create a copy of the dataset for Host Analysis ONLY
    df_host_analysis = df_popin.copy()

    # Ensure Hosted_By column exists to prevent errors
    if "Hosted_By" in df_host_analysis.columns:

        def create_top_hosts_chart(num_hosts):
            """Creates a bar chart of top event hosts by number of events."""
            top_hosts = df_host_analysis['Hosted_By'].value_counts().nlargest(num_hosts).reset_index()
            top_hosts.columns = ['Hosted_By', 'Number_of_Events']  # Rename columns

            # Get events hosted by each host
            events_hosted = df_host_analysis.groupby('Hosted_By')['Event_Name'].apply(list).reset_index()
            events_hosted.columns = ['Hosted_By', 'Events_Hosted']  # Rename columns

            # Merge to get events hosted into top_hosts DataFrame
            top_hosts = pd.merge(top_hosts, events_hosted, on='Hosted_By')

            # Create chart
            top_hosts_chart = alt.Chart(top_hosts).mark_bar().encode(
                x=alt.X("Hosted_By:N", sort='-y', title="Host"),  
                y=alt.Y("Number_of_Events:Q", title="Number of Events"),  
                tooltip=["Hosted_By", "Number_of_Events", "Events_Hosted"],  
                color=alt.Color("Hosted_By:N", scale=alt.Scale(scheme='category10'))
            ).properties(title="Top Event Hosts", width=700, height=400)
            return top_hosts_chart

        def create_hosts_attendance_chart(num_hosts):
            """Creates a bar chart of hosts with the highest attendance."""
            top_attendees_hosts = df_host_analysis.groupby('Hosted_By')['Attendees'].sum().nlargest(num_hosts)
            hosts_attendance_chart = alt.Chart(top_attendees_hosts.reset_index()).mark_bar().encode(
                x=alt.X("Hosted_By:N", sort='-y', title="Host"),  
                y=alt.Y("Attendees:Q", title="Total Attendees"),
                tooltip=["Hosted_By", "Attendees"],
                color=alt.Color("Hosted_By:N", scale=alt.Scale(scheme='category10'))
            ).properties(title="Hosts with the Highest Attendance", width=700, height=400)
            return hosts_attendance_chart

        # --- Top Event Hosts Section ---
        st.subheader("Top Event Hosts")  
        num_hosts_top_events = st.slider("Select number of top event hosts to display:", min_value=5, max_value=20, value=10, key="top_events_slider") 
        top_hosts_chart = create_top_hosts_chart(num_hosts_top_events)
        st.altair_chart(top_hosts_chart, use_container_width=True) 

        # --- Hosts with the Highest Attendance Section ---
        st.subheader("Hosts with the Highest Attendance")  
        num_hosts_attendance = st.slider("Select number of hosts with highest attendance to display:", min_value=5, max_value=20, value=10, key="attendance_slider")
        hosts_attendance_chart = create_hosts_attendance_chart(num_hosts_attendance) 
        st.altair_chart(hosts_attendance_chart, use_container_width=True)

    else:
        st.warning("No 'Hosted_By' column found in dataset. Please check the data source.")

# ------------------ END FOR HOST ANALYSIS --------------------------- #

# ----------------- EVENT LOCATION INSIGHTS ----------------- #
if st.session_state.graph_selection == "Event Location Insights":
    st.subheader("📍 Interactive Map of Events")

    # ADD EVENT ID COLUMN if not present
    if "Event_ID" not in df_popin.columns:
        df_popin["Event_ID"] = df_popin.index + 1

    location_events = df_popin.groupby(["latitude", "longitude"]).agg({
        "Event_Name": lambda x: "<br>".join(x),
        "Location": "first",
        "Event_ID": "count"
    }).reset_index()

    m_events = folium.Map(location=[df_popin["latitude"].mean(), df_popin["longitude"].mean()], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m_events)

    for _, row in location_events.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            popup_content = f"<b>Location:</b> {row['Location']}<br><b>Events:</b><br>{row['Event_Name']}"
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup_content,
                tooltip=f"Events: {row['Event_ID']}",
            ).add_to(marker_cluster)

    st.components.v1.html(m_events.get_root().render(), height=600)

    #--------------MICA'S CODE----------------#
    # Location: Attendees vs Events

    # Ensure "Location" mapping is available
    location_map = df_popin.groupby(["latitude", "longitude"])["Location"].first().to_dict()

    # Get Top 10 Locations with Most Attendees
    if "Attendees" in df_popin.columns:
        top_attended_locations = (
            df_popin.groupby(["latitude", "longitude"])["Attendees"]
            .sum()
            .reset_index()
            .sort_values(by="Attendees", ascending=False)
            .head(10)
        )

        # Add the Location name (if available) or "Unknown Location"
        top_attended_locations["Location"] = top_attended_locations.apply(
            lambda row: location_map.get((row["latitude"], row["longitude"]), "Unknown Location"), axis=1
        )

    # Get Top 10 Locations with Most Events
    top_event_locations = (
        df_popin.groupby(["latitude", "longitude"])
        .size()
        .reset_index(name="Event Count")  # Create "Event Count" column
        .sort_values(by="Event Count", ascending=False)
        .head(10)
    )

    # Add the Location name (if available) or "Unknown Location"
    top_event_locations["Location"] = top_event_locations.apply(
        lambda row: location_map.get((row["latitude"], row["longitude"]), "Unknown Location"), axis=1
    )

    # Create a unified list of unique top locations from both lists
    unique_locations = pd.concat([top_attended_locations["Location"], top_event_locations["Location"]]).unique()

    # Filter the original dataset to include only these locations
    filtered_df = df_popin[df_popin["Location"].isin(unique_locations)]

    # Compute total event count & attendees for these locations
    final_merged_df = (
        filtered_df.groupby("Location")
        .agg({"Attendees": "sum"})  # Sum attendees
        .reset_index()
    )

    # Add Event Count manually since it wasn't included in agg()
    event_counts = (
        filtered_df.groupby("Location")
        .size()
        .reset_index(name="Event Count")
    )

    # Compute Additional Statistics
    attendee_stats = (
        filtered_df.groupby("Location")["Attendees"]
        .agg(["median", "max", "min"])
        .reset_index()
        .rename(columns={"median": "Median Attendees per Location", "max": "Max Attendees per Location", "min": "Min Attendees per Location"})
    )

    # Merge event count and attendee statistics into the final dataframe
    final_merged_df = pd.merge(final_merged_df, event_counts, on="Location", how="left")
    final_merged_df = pd.merge(final_merged_df, attendee_stats, on="Location", how="left")

    # Fix "Min Attendees per Location"
    # If a location has only 1 event, Min Attendees = Max Attendees
    final_merged_df.loc[final_merged_df["Event Count"] == 1, "Min Attendees per Location"] = final_merged_df["Max Attendees per Location"]

    # Compute Average Attendees per Event
    final_merged_df["Average Attendees per Event"] = final_merged_df["Attendees"] / final_merged_df["Event Count"]
    final_merged_df["Average Attendees per Event"] = final_merged_df["Average Attendees per Event"].round(2)  # Round to 2 decimal places

    # Remove latitude and longitude from the dataset completely
    final_merged_df = final_merged_df[[
        "Location", "Attendees", "Event Count", 
        "Average Attendees per Event", "Median Attendees per Location", 
        "Max Attendees per Location", "Min Attendees per Location"
    ]]

    # Sort locations by Total Attendees (Descending)
    final_merged_df = final_merged_df.sort_values(by="Attendees", ascending=False)

    # Convert to long format for Altair (for line chart)
    chart_data = final_merged_df.melt(id_vars=["Location"], var_name="Metric", value_name="Value")

    # Filter the chart data to include only relevant metrics
    chart_data_filtered = chart_data[chart_data["Metric"].isin([
        "Attendees", "Average Attendees per Event", "Max Attendees per Location", "Min Attendees per Location"
    ])]

    # STREAMLIT DASHBOARD
    st.title("📊 Top 10 Locations: Events vs Attendees")

    # Display Sorted Table
    st.dataframe(final_merged_df)

    # Create a Line Chart 
    line_chart = alt.Chart(chart_data_filtered).mark_line(point=True).encode(
        x=alt.X("Location:N", sort=final_merged_df["Location"].tolist(), title="Location"),  # Sort by attendees
        y=alt.Y("Value:Q", title="Count"),
        color=alt.Color("Metric:N", scale=alt.Scale(scheme="category10"), title="Metric"),
        tooltip=["Location", "Metric", "Value"]
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)

#----------------END OF MICA'S CODE---------------#

# ----------------- WORD CLOUD ----------------- #
elif st.session_state.graph_selection == "Word Cloud":
    st.subheader("🗣️ Word Cloud Analysis")

    # ---  Most Common Words in All Event Titles ---
    st.subheader("☁ Most Common Words in Event Titles")
    text_data = ' '.join(df_popin['Event_Name'].dropna())
    text_data = re.sub(f"[{string.punctuation}]", "", text_data)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    #------------Marla--------------------#

    # ---  Most Common Words in 'Other' Events ---
    st.subheader("📋 Common Words in 'Other' Events")

    df_others = df_popin[df_popin["Event_Category"] == "Other"].copy()
    if df_others.empty:
        st.warning("No events found in the 'Other' category.")
    else:
        words_other = " ".join(df_others["Event_Name"].dropna())
        wordcloud_other = WordCloud(width=800, height=400, background_color='white').generate(words_other)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_other, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    # --- Suggested Topics for 'Other' Events ---  NOTE: DECIDING TO DO THIS ONE IN CATEGORY ANALYSIS INSTEAD
    st.subheader("🔍 Suggested Topics for 'Other' Events")

    if df_others.empty:
        st.warning("No events found in the 'Other' category.")
    else:
        # Initialize text processing
        vectorizer = CountVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(df_others["Event_Name"].dropna())

        # Perform Latent Dirichlet Allocation (LDA) for topic modeling
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = {f"Topic {i+1}": [feature_names[j] for j in topic.argsort()[:-8:-1]] for i, topic in enumerate(lda.components_)}

        # Display suggested topics
        st.write("### Suggested New Topics from 'Other' Events")
        for topic, words in topics.items():
            st.write(f"**{topic}:** {', '.join(words)}")

    # ---  Most Common Words in Online Events ---
    st.subheader("☁ Most Common Words in Online Event Titles")

    df_online = df_popin[df_popin["Event_Type"] == "Online"]
    if df_online.empty:
        st.warning("No online events found.")
    else:
        text_data_online = ' '.join(df_online['Event_Name'].dropna())
        text_data_online = re.sub(f"[{string.punctuation}]", "", text_data_online)

        wordcloud_online = WordCloud(width=800, height=400, background_color='white').generate(text_data_online)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_online, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    # ---  Most Common Words in In-Person Events ---
    st.subheader("☁ Most Common Words in In-Person Event Titles")

    df_in_person = df_popin[df_popin["Event_Type"] == "In-Person"]
    if df_in_person.empty:
        st.warning("No in-person events found.")
    else:
        text_data_in_person = ' '.join(df_in_person['Event_Name'].dropna())
        text_data_in_person = re.sub(f"[{string.punctuation}]", "", text_data_in_person)

        wordcloud_in_person = WordCloud(width=800, height=400, background_color='white').generate(text_data_in_person)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_in_person, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)



#-------------------------- End of Marla's------------------#
