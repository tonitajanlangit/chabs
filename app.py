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
import string 
import googlemaps
import time
from folium.plugins import MarkerCluster



st.set_page_config(
    page_title="PopIn Data Analysis Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom Altair color scheme
PRIMARY_COLOR = "#fc6c64"  # Coral Red
WHITE = "#ffffff"  # White
COLORS = ["#3aafa9", "#fc6c64", "#f4d03f"] # Chart colors
PASTEL = ["#ff9999", "#dda0dd", "#20b2aa"] # For variation

# Load PopIn dataset
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
            .main {{ background-color: white; }}
            [data-testid="stSidebar"] {{ background-color: {PRIMARY_COLOR}; }}
            h1, h2, h3, h4, h5, h6, p, label {{ color: black !important; }}
            div[data-testid="stMetric"] {{ background-color: {PRIMARY_COLOR}; padding: 10px; border-radius: 10px; color: white; text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True
    )
apply_custom_styles()

def apply_custom_styles():
    st.markdown(
        f"""
        <style>
            .main {{ background-color: white; }}
            [data-testid="stSidebar"] {{ background-color: {PRIMARY_COLOR}; }}
            h1, h2, h3, h4, h5, h6, p, label {{ color: black !important; }}
            div[data-testid="stMetric"] {{ background-color: {PRIMARY_COLOR}; padding: 10px; border-radius: 10px; color: white; text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True
    )
apply_custom_styles()

st.markdown("""
    <style>
        .button-container {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
        }
        .button-container button {
            width: 180px;
            height: 40px;
            font-size: 14px;
            border-radius: 5px;
            background-color: #fc6c64;
            color: white;
            cursor: pointer;
            text-align: center;
        }
        .button-container button:hover {
            background-color: #ff5733;
        }
        .selected-button {
            background-color: #ff5733 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation buttons
with st.sidebar:
    st.title('PopIn Data Analysis Filters')
    event_category = st.selectbox("Select Category", ["All", "Business", "Entertainment", "Other"])
    
    st.subheader("Select Visualization")
    event_buttons = [
        "Event Performance Overview",
        "Category Analysis",
        "Event Trends Over Time",
        "Event Popularity",
        "Online vs. In-Person Events",
        "Event Location Insights",
        "Word Cloud"
    ]
    
    # Initialize session state for button selection
    if 'graph_selection' not in st.session_state:
        st.session_state.graph_selection = event_buttons[0]
    
    # Create buttons with persistent selection state
    for button in event_buttons:
        if st.button(button, key=button):
            st.session_state.graph_selection = button

# Apply styles to highlight selected button
st.markdown(
    f"""
    <style>
        .stButton > button {{
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
            background-color: #fc6c64;
            color: white;
            cursor: pointer;
        }}
        .stButton > button:hover {{
            background-color: #ff5733;
        }}
        .stButton > button[selected] {{
            background-color: #ff5733 !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display selected visualization
st.subheader(f"Selected View: {st.session_state.graph_selection}")

if event_category != "All":
    df_popin = df_popin[df_popin['Category'] == event_category]

if 'Date_&_Time' in df_popin.columns:
    df_popin['Date'] = df_popin['Date_&_Time'].str.split('T').str[0]
    df_popin['Time'] = df_popin['Date_&_Time'].str.split('T').str[1]
    df_popin.drop(columns=['Date_&_Time'], inplace=True)

df_popin['Date'] = pd.to_datetime(df_popin['Date'], errors='coerce')
df_popin['Month-Year'] = df_popin['Date'].dt.to_period('M')


#EVENT PERFORMANCE OVERVIEW HERE

if st.session_state.graph_selection == "Event Performance Overview":
  st.subheader("📊 Event Performance Overview")
  total_events = len(df_popin)
  total_attendees = df_popin['Attendees'].sum() if 'Attendees' in df_popin.columns else 0
  col1, col2 = st.columns(2)
  with col1:
        st.metric(label="🏆 Total Events", value=total_events)
  with col2:
        st.metric(label="👥 Total Attendees", value=total_attendees)
   
  st.subheader("📈 Average Attendance Per Event Category") 
    
  if not df_popin.empty:
        # Calculate average attendance
        average_attendance = df_popin.groupby('Category')['Attendees'].mean().reset_index()
        average_attendance.columns = ['Category', 'Average Attendance']

        # Create bar chart for average attendance
        avg_attendance_chart = alt.Chart(average_attendance).mark_bar().encode(
            x=alt.X("Average Attendance:Q", title="Average Attendance"),
            y=alt.Y("Category:N", sort='-x', title="Category"),
            tooltip=["Category", "Average Attendance"],
            color=alt.Color("Category:N", scale=alt.Scale(domain=average_attendance['Category'].tolist(), range=COLORS)),
            text=alt.Text("Average Attendance:Q", format=".0f")  # Display the average attendance next to bars
        ).properties(width=750, height=400)

        # Create text annotations for average attendance
        avg_annotations = alt.Chart(average_attendance).mark_text(
            align='left',
            dx=10, dy=-10,  # Adjusting the position slightly above each bar
            fontSize=12,
            fontWeight='bold',
            color='black'
        ).encode(
            x=alt.X('Average Attendance:Q', title=None),
            y=alt.Y('Category:N', sort='-x', title=None),
            text=alt.Text('Average Attendance:Q', format='.0f')  # Add the average attendance as labels
        )

        # Combine the bar chart with the annotations (without the config causing issues)
        avg_final_chart = avg_attendance_chart + avg_annotations

        # Configure the final chart's style and appearance
        avg_final_chart = avg_final_chart.configure_mark(opacity=0.8)

        # Display the Average Attendance per Event chart
        st.altair_chart(avg_final_chart, use_container_width=True)

  else:
        st.warning("No data available for the selected category.")

#DONUT CHART ONLINE VS INPERSON

# 🎯 1. Categorize Locations as "Online" or "In-Person"
df_popin["Location Type"] = df_popin["Location"].apply(lambda x: "Online" if x.lower() == "online" else "In-Person")

# 🎯 2. Count Online vs. In-Person Events and Compute Percentages
location_counts = df_popin["Location Type"].value_counts().reset_index()
location_counts.columns = ["Location Type", "Count"]
location_counts["Percentage"] = (location_counts["Count"] / location_counts["Count"].sum()) * 100

# 🎨 3. Function to Create a Properly Centered Donut Chart
def make_donut_chart(input_df, category_col, value_col, percentage_col, title):
    # Base donut chart
    chart = alt.Chart(input_df).mark_arc(innerRadius=50, cornerRadius=15).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(f"{category_col}:N", scale=alt.Scale(scheme="category10")),  # Netflix-style colors
        tooltip=[category_col, value_col]  # Shows counts on hover
    )

    # Properly center percentage labels on arcs
    text = alt.Chart(input_df).mark_text(size=14, fontWeight="bold", color="white").encode(
        theta=alt.Theta(f"{value_col}:Q"),
        text=alt.Text(f"{percentage_col}:Q", format=".1f"),  # Display percentage with 1 decimal
        color=alt.Color(f"{category_col}:N", scale=alt.Scale(scheme="category10")),  # Same color as slices
        radius=alt.value(80)  # Moves text outward from center of the donut
    )

    return (chart + text).properties(width=300, height=300, title=title)

# 🎥 4. Display Donut Chart in Streamlit
st.subheader("📍 Online vs. In-Person Events")
st.altair_chart(make_donut_chart(location_counts, "Location Type", "Count", "Percentage", "Event Distribution"), use_container_width=True)


#CATEGORY ANALYSIS HERE

if st.session_state.graph_selection == "Category Analysis":
    st.subheader("🎭 Event Count by Category")

    if not df_popin.empty:
        # Event count per category
        event_count = df_popin['Category'].value_counts().reset_index()
        event_count.columns = ['Category', 'Count']
        
        # Add a new column for color based on category
        event_count['Color'] = event_count['Category'].map(
            lambda x: COLORS[df_popin['Category'].unique().tolist().index(x)]
        )
        
        # Create the first base chart (Horizontal bar chart)
        chart = alt.Chart(event_count[:20]).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of Events"),
            y=alt.Y("Category:N", sort='-x', title="Category"),
            tooltip=["Category", "Count"],
            color=alt.Color("Category:N", scale=alt.Scale(domain=event_count['Category'].tolist(), range=COLORS)),
            text=alt.Text("Count:Q", format=".0f")  # Display the count as text next to bars
        ).properties(width=750, height=400)

        # Create text annotations for each category
        annotations = alt.Chart(event_count).mark_text(
            align='left',
            dx=10, dy=-10,  # Adjusting the position slightly above each bar
            fontSize=12,
            fontWeight='bold',
            color='black'
        ).encode(
            x=alt.X('Count:Q', title=None),
            y=alt.Y('Category:N', sort='-x', title=None),
            text=alt.Text('Count:Q', format='.0f')  # Add the count of events as labels
        )

        # Combine the bar chart with the annotations
        final_chart = chart + annotations

        # Configure the final chart's style and appearance
        final_chart = final_chart.configure_mark(opacity=0.8)

        # Display the final chart
        st.altair_chart(final_chart, use_container_width=True)

        # --- New Graph for Average Attendees per Category (Vertical Layout) ---
        st.subheader("📈 Average Attendees by Category")

        # Calculate average attendance per category
        average_attendance = df_popin.groupby('Category')['Attendees'].mean().reset_index()
        average_attendance.columns = ['Category', 'Average Attendance']

        #Create a VERTICAL bar chart for Average Attendance
        avg_attendance_chart = alt.Chart(average_attendance).mark_bar(cornerRadius=4).encode(
            y=alt.Y("Category:N", title="Category", sort='-x'),  # Category on y-axis (vertical bars)
            x=alt.X("Average Attendance:Q", title="Average Attendance"),  # Average Attendance on x-axis
            tooltip=["Category", "Average Attendance"],
            color=alt.Color("Category:N", scale=alt.Scale(domain=average_attendance['Category'].tolist(), range=PASTEL)),
            text=alt.Text("Average Attendance:Q", format=".0f")  # Display the average attendance inside the bars
        ).properties(width=750, height=400)

        #Create text annotations *outside* the bars
        avg_annotations = alt.Chart(average_attendance).mark_text(
            align='left',
            dx=10, dy=0,  # Adjust to place text outside the bars
            fontSize=12,
            fontWeight='bold',
            color='black'
        ).encode(
            x=alt.X('Average Attendance:Q', title=None),
            y=alt.Y('Category:N', sort='-x', title=None),
            text=alt.Text('Average Attendance:Q', format='.0f')  # Add the average attendance outside bars
        )

        # Combine the vertical bar chart with the annotations
        avg_final_chart = avg_attendance_chart + avg_annotations

        # Display the average attendance chart
        st.altair_chart(avg_final_chart, use_container_width=True)

    else:
        st.warning("No data available for the selected category.")

# EVENT POPULARITY HERE
elif st.session_state.graph_selection == "Event Popularity":
    st.subheader("⭐ Top 10 Most Attended Events")

    if "Attendees" in df_popin.columns and not df_popin["Attendees"].isna().all():
        # Get top 10 events
        top_attended = df_popin.nlargest(10, "Attendees")[["Event_Name", "Attendees"]]

        # 🎨 *Stacked Bar Chart*
        bar_chart = (
            alt.Chart(top_attended)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Attendees:Q", title="Attendees"),
                y=alt.Y("Event_Name:N", title="Event Name", sort='-x'),
                tooltip=["Event_Name", "Attendees"],
                color=alt.Color("Attendees:Q", scale=alt.Scale(scheme='reds')),
                text=alt.Text("Attendees:Q", format=".0f")  # Add attendee counts as text
            )
            .properties(width=750, height=400)
        )

        bar_chart = bar_chart.configure_mark(opacity=0.8)

        # Display chart
        st.altair_chart(bar_chart, use_container_width=True)
 
# 🚨 *Events with the Lowest Attendance*
        st.subheader("🚨 Events with the Lowest Attendance")

        # Get bottom 10 events by attendance
        lowest_attended = df_popin.nsmallest(10, "Attendees")[["Event_Name", "Attendees"]]

        # 🎨 *Stacked Bar Chart for Lowest Attendance*
        low_attendance_chart = (
            alt.Chart(lowest_attended)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Attendees:Q", title="Number of Attendees"),
                y=alt.Y("Event_Name:N", title=None, sort="x"),
                color=alt.Color("Attendees:Q", scale=alt.Scale(scheme="reds")),
                tooltip=["Event_Name", "Attendees"]
            )
            .properties(width=750, height=400)
        )

        # 📌 *Data Labels for Lowest Attendance*
        low_text_labels = (
            alt.Chart(lowest_attended)
            .mark_text(align="left", dx=5, fontSize=12, color="black", fontWeight="bold")
            .encode(
                x=alt.X("Attendees:Q"),
                y=alt.Y("Event_Name:N", sort="x"),
                text=alt.Text("Attendees:Q", format=".0f")
            )
        )

        # 🔹 *Combine the Lowest Attendance Chart & Labels*
        low_final_chart = low_attendance_chart + low_text_labels

        # 🚀 *Display the Lowest Attendance Chart*
        st.altair_chart(low_final_chart, use_container_width=True)

    else:
        st.warning("⚠ No valid attendee data available.")


#EVENT TRENDS OVERTIME HERE
if st.session_state.graph_selection == "Event Trends Over Time":
    st.subheader("📈 Event Trends Over Time")

# 📅 Line Chart for Events Over Time

#df_popin["Date"] = df_popin["Date"].dt.date
df_popin["Date"] = pd.to_datetime(df_popin["Date"], errors='coerce').dt.date
events_over_time = df_popin.groupby("Date").size().reset_index(name="Event_Count")
line_events = alt.Chart(events_over_time).mark_line().encode(
    x=alt.X("Date:T", title="Date"),
    y=alt.Y("Event_Count:Q", title="Number of Events"),
    tooltip=["Date", "Event_Count"]
).properties(title="Event Trends Over Time", width=700, height=400)

st.altair_chart(line_events, use_container_width=True)



# EVENT LOCATION INSIGHTS HERE
#1. EVENT MAP

if st.session_state.graph_selection == "Event Location Insight":
    st.subheader("📍 Event Map")

###ADDING EVENT ID COLUMN
if "Event_ID" not in df_popin.columns:
    df_popin["Event_ID"] = df_popin.index + 1  # Assigns a unique ID to each event

###CLEANING LOCATION COLUMN

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

### --- 📌 GEOCODING (Convert Location to Latitude & Longitude) --- ###

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

### --- 📌 STREAMLIT DASHBOARD (Map Visualization) --- ###

st.title("📍 Event Locations on Map")

# Ensure latitude and longitude are numeric
df_popin["latitude"] = pd.to_numeric(df_popin["latitude"], errors="coerce")
df_popin["longitude"] = pd.to_numeric(df_popin["longitude"], errors="coerce")

# Aggregate event names and count for each location
location_events = df_popin.groupby(["latitude", "longitude"]).agg({
    "Event_Name": lambda x: "<br>".join(x),  # Concatenate event names with line breaks
    "Location": "first",
    "Event_ID": "count"  # Count number of events at the same location
}).reset_index()

# Create a Folium map centered at the mean latitude and longitude
m_events = folium.Map(location=[df_popin["latitude"].mean(), df_popin["longitude"].mean()], zoom_start=5)

# Create a marker cluster
marker_cluster = MarkerCluster().add_to(m_events)

# Add markers for each event location
for _, row in location_events.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        popup_content = f"<b>Location:</b> {row['Location']}<br><b>Events:</b><br>{row['Event_Name']}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_content,
            tooltip=f"Events: {row['Event_ID']}",  # Show event count when hovering
        ).add_to(marker_cluster)

# Show the map in Streamlit
st.write("### Interactive Map of Events")
st.components.v1.html(m_events.get_root().render(), height=600)

#Display map using streamlit_folium
    #st_folium(m_events, width=725)


# Define a dictionary to map categories to colors
category_colors = {
    'Business': 'blue',
    'Entertainment': 'red',
    'Others': 'green'
}


  #WORD CLOUD HERE
if st.session_state.graph_selection == "Word Cloud":  # Access graph_selection from session state
    st.subheader("🗣 Word Cloud")

    text_data = ' '.join(df_popin['Event_Name'].dropna())
    
    # Remove unwanted characters
    text_data = re.sub(f"[{string.punctuation}]", "", text_data)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # Display word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
