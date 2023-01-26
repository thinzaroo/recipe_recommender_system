import streamlit as st
from classes.User import User
import utils

def update_gender():
    st.session_state.gender = st.session_state.gender

def update_height():
    st.session_state.height = st.session_state.height

def update_weight():
    st.session_state.weight = st.session_state.weight

def update_health_cond():
    st.session_state.health_cond = st.session_state.health_cond

def print_sidebar_from_session():
    if st.session_state.age != '0':
        gender = st.session_state.gender
        age = st.session_state.age
        height = st.session_state.height
        weight = st.session_state.weight
        activity_level_index = st.session_state.activity_lvl_index
        health_condition = st.session_state.health_condition

        st.sidebar.header("Your Profile")
        st.sidebar.text('Gender:' + gender)
        st.sidebar.text('Age:' + age)
        st.sidebar.text('Height:' + height)
        st.sidebar.text('Weight:'+ weight)
        st.sidebar.text('Activity Level:' + utils.activity_level_options[activity_level_index])
        st.sidebar.text('Health condition:' + health_condition)
        
        new_user = User(gender, int(age), float(weight), float(height), int(activity_level_index), health_condition)
        st.sidebar.text('BMI:' + str(new_user.calculate_bmi()))
        st.sidebar.text('You are ' + new_user.get_bmi_category())
        st.sidebar.text('BMR: ' + str(new_user.calculate_bmr()))
        st.sidebar.text('Recommended daily calorie: ' + str(new_user.get_recommended_daily_calorie()))

def main():
    st.header("Recipe Recommender System")

    # initialize session state variables
    if 'activity_lvl_index' not in st.session_state:
        st.session_state.activity_lvl_index = 0

    if 'health_condition' not in st.session_state:
        st.session_state.health_condition = 'none'

    if 'gender' not in st.session_state:
        st.session_state.gender = 'Male'
    
    if 'age' not in st.session_state:
        st.session_state.age = 0

    #======= Capture User Profile =======
    st.markdown("### User Profile")

    # Gender - radio
    gender = st.radio(
        "Gender:",
        ('Male', 'Female'),
        key='gender', 
        on_change=update_gender
    )

    # Let's put Age, Height, Weight in columns
    col1, col2, col3 = st.columns(3)
    
    # Age - input text
    age = col1.text_input(label="Age", value=30)
    if age:
        st.session_state.age = age

    # height in cm - input text
    height = col2.text_input(label="Height", value=170)
    if height:
        st.session_state.height = height

    # weight in kg - input text
    weight = col3.text_input(label="Weight", value=65)
    if weight:
        st.session_state.weight = weight

    col4, col5 = st.columns(2)

    # Activity level - select box
    activity_level_index = col4.selectbox("Your activity Level:", range(len(utils.activity_level_options)), format_func=lambda x: utils.activity_level_options[x])
    if activity_level_index:
        st.session_state.activity_lvl_index = activity_level_index
    
    # Health condition 
    health_condition = col5.selectbox(
        "Health condition:",
        (
            "Healthy",
            "Diabetic",
            "Overweight",
            "Hypertension"
        )
    )
    if health_condition:
        st.session_state.health_condition = health_condition

    # Generate Report
    btn_generate_clicked = st.button("Generate Report")
    if btn_generate_clicked:
        print_sidebar_from_session()

if __name__ == '__main__':
    main()