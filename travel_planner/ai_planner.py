import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from datetime import date

# Load API Key
try:
    with open("API_key.txt", "r") as f:
        api_key = f.read().strip()
except Exception as e:
    st.error("API Key Error! Please check your API key file.")
    st.stop()

# Configure LangChain Model
chat_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash-exp")
output_parser = StrOutputParser()

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="centered")

st.markdown("""
    <style>
        .title { text-align: center; font-size: 48px; font-weight: bold; color: #1E90FF; }
        .subtitle { text-align: center; font-size: 18px; color: #666; }
        .stButton>button { background-color: #1E90FF; color: white; font-size: 16px; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# Header
# st.title("AI-Powered Travel Planner")
st.markdown("<div class='title'>AI-Powered Travel Planner</div>", unsafe_allow_html=True)
st.markdown(' ')

col1, col2 = st.columns(2)
with col1:
    source = st.text_input("Enter Source Location", placeholder="E.g., Hyderabad").strip()
with col2:
    destination = st.text_input("Enter Destination", placeholder="E.g., Kathmandu").strip()

travel_date = st.date_input("Select Travel Date", min_value=date.today())

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("Number of Days", min_value=1, max_value=30, value=3)
with col2:
    passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)

time_preference = st.selectbox("Preferred Travel Time", ["Anytime", "Morning", "Afternoon", "Evening", "Night"])
travel_modes = st.multiselect("Select Preferred Travel Modes", ["Cab", "Train", "Bus", "Flight"], default=["Flight", "Train"])
preference = st.radio("Priority", ["Cheapest", "Fastest", "Best Balance"])

# Prompt Template
prompt = ChatPromptTemplate(messages=
    [('system', """You are an AI travel planner that provides users with personalized travel plans based on their input. 
      Given details such as departure location, destination, travel dates, preferred transport, and accommodation preferences, 
      generate a clear and concise travel itinerary with estimated costs.

    Instructions for the AI:
    - Analyze the user's input and determine the best travel options (flights, trains, buses, or cabs).
    - Estimate the total cost, including transportation, accommodation, and key expenses (e.g., food, activities).
    - Provide two optimized travel options, balancing affordability, convenience, and travel time.
    - Keep the response clear and short, ensuring the user can easily understand the itinerary and costs.
    - Break down the itinerary into key components: Transport, Stay, Cost Summary, Best Choice Recommendation, and a Day-wise Travel Plan.
    - Suggest the best places to visit at the destination, ensuring a well-structured itinerary.
    - Explain cost estimations briefly, ensuring they are realistic.

    Disclaimer: This is an estimated plan based on current information and projections. Actual costs and availability may vary. It's 
      essential to conduct your own research and book your travel arrangements well in advance."""),
    ('human', """User Input:
    Source: {source}
    Destination: {destination}
    Travel Date: {date}
    Days: {days}
    Preference: {preference}
    Time Preference: {time_preference}
    Modes: {travel_modes}
    Passengers: {passengers}
    """)
    ])

if st.button("Get Travel Options"):
    if not source or not destination:
        st.error("Please enter both source and destination!")
    else:
        with st.spinner(f"Searching travel options from **{source}** to **{destination}** on {travel_date} for {days} days..."):
            user_input = {"source": source, "destination": destination, "date": str(travel_date), "days": days, "preference": preference, 
                          "time_preference": time_preference, "travel_modes": travel_modes, "passengers": passengers}
            mod = prompt | chat_model | output_parser
            result = mod.invoke(user_input)
            st.subheader("Recommended Travel Options:")
            st.write(result)

