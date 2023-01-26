import streamlit as st
from gensim.models import Word2Vec

import pandas as pd
from numpy import dot
import random
from numpy.linalg import norm

from classes.User import User
import utils
from time import time

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

def find_Similar_dish(df, recipe_index,embedding_to_use):
    #ingredients_embedding_w2v

    a = df.loc[recipe_index, embedding_to_use]
    orn = df.loc[recipe_index, "title"]
    
    dishtances = {}
    for i in range(len(df)):
        if i==recipe_index:
            continue;
        try:
            dn = df.loc[i, "title"]
            b = df.loc[i, embedding_to_use]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            if cos_sim not in dishtances.values():
                dishtances[i] = cos_sim
        except:
            continue;
            
    dishtances_2 = {k: v for k, v in sorted(dishtances.items(), key=lambda item: item[1], reverse = True)}
    mostSimilarDishes = []
    countSim = 0
    for el in dishtances_2.keys():
        mostSimilarDishes.append(el)
        countSim+=1
        if countSim==10:
            break;
    return mostSimilarDishes

def list_Similar_dishes(df, xx, embeddingToUse):
    
    additionalColumns = ['nutr_values_per100g_energy','ingredients_simple','url','nutr_values_per100g_protein']
    similarList1 = find_Similar_dish(df,xx,embeddingToUse)
    simResults1 = []

    allSuggestedDishNames = []

    for simIndex in similarList1:
        tempRes = []
        dName = df.loc[simIndex, "title"]
        dishName = " ".join([w for w in dName.split() if w.lower()!='recipe'])
        tempRes.append(dishName)
        dishNameShort = " ".join(dishName.split()[-2:])
        allSuggestedDishNames.append(dishNameShort)
        for col in additionalColumns:
            tempRes.append(df.loc[simIndex, col])
        simResults1.append(tempRes)
    
    additionalColumns.insert(0,"Dish")
    
    return(pd.DataFrame(simResults1, columns = additionalColumns),allSuggestedDishNames)

def main():
    df_recipes = pd.read_pickle('processed/df_recipes_with_embeddings.pkl')

    #======= Recommendation based on ingredients =======
    st.markdown("### Recommend similar recipe")
    
    arr_recipe_list = ("Chicken Wings", 
        "Onion Soup Mix", 
        "West African Chicken Peanut Soup",
        "Double Chocolate Pudding",
        "California Chicken Salad",
        "Mango Lassi",
        "Creamed Broccoli And Cauliflower Soup",
        "Honey Chive Codfish",
        "Thai Grilled Beef"
        )
    recipe_index = st.radio("Select an item to generate similar recipe:", range(len(arr_recipe_list)), format_func=lambda x: arr_recipe_list[x])
    if recipe_index == 0:
        recipe_id = 786
    elif recipe_index == 1:
        recipe_id = 32317
    elif recipe_index == 2:
        recipe_id = 111
    elif recipe_index == 3:
        recipe_id = 3333
    elif recipe_index == 4:
        recipe_id = 1859
    elif recipe_index == 5:
        recipe_id = 19
    elif recipe_index == 6:
        recipe_id = 11
    elif recipe_index == 7:
        recipe_id = 22036
    elif recipe_index == 8:
        recipe_id = 2382
    else:
        recipe_id = 33

    btn_recommend2_clicked = st.button('Recommend me')
    
    if btn_recommend2_clicked:
        st.text("Here is a list of similar recipes similar to: " + df_recipes.loc[recipe_id]['title'])
        st.spinner()
        with st.spinner(text="Loading..."):
            t = time()
            res = list_Similar_dishes(df_recipes, recipe_id, "ingredients_embedding_w2v")    
            st.dataframe(res[0])
            st.text('\nTime taken: {} mins'.format(round((time() - t) / 60, 2)))

if __name__ == '__main__':
    main()
    #print_sidebar_from_session()