import streamlit as st
from gensim.models import Word2Vec
import unidecode
import ast

import numpy as np
import pandas as pd
from time import time  # To log the operations time

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from ingredient_parser import ingredient_parser
from classes.MeanEmbeddingVectorizer import MeanEmbeddingVectorizer
from classes.TfidfEmbeddingVectorizer import TfidfEmbeddingVectorizer

from classes.User import User
import utils

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

def get_and_sort_corpus(data):
    """
    Get corpus with the documents sorted in alphabetical order
    """
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def get_recommendations(N, scores):
    """
    Top-N recomendations order by score
    """
    # load in recipe dataset
    df_recipes = pd.read_csv('data/recipes_feature_engineered_test.csv')
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["title", "score", "calorie", "ingredients",  "url"])
    count = 0
    for i in top:
        recommendation.at[count, "title"] = title_parser(df_recipes["title"][i])
        recommendation.at[count, "calorie"] = df_recipes["nutr_values_per100g_energy"][i]
        recommendation.at[count, "ingredients"] = ingredient_parser_final(
            df_recipes["ingredients_simple"][i]
        )
        recommendation.at[count, "url"] = df_recipes["url"][i]
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title


def ingredient_parser_final(ingredient):
    """
    neaten the ingredients being outputted
    """
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def get_recs(ingredients, N=5, mean=False):
    # load in word2vec model
    model = Word2Vec.load("models/fasttext_model_cbow_300.bin")
    # model = Word2Vec.load("models/w2v_model_cbow_300.bin")
    #model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")
    # load in data
    data = pd.read_csv('data/recipes_feature_engineered_test.csv')
    # parse ingredients
    data["parsed"] = data.ingredients_simple.apply(ingredient_parser)
    # create corpus
    corpus = get_and_sort_corpus(data)

    if mean:
        # get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        # use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    # create embessing for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    # parse ingredient list
    input = ingredient_parser(input)
    # get embeddings for ingredient doc
    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations

def main():
    
    # load model
    model = Word2Vec.load('./models/w2v_model_for_embedding.bin')
    words = list(model.wv.index_to_key)
    words.sort()
    
    # UI components
    st.markdown("### Recommend Recipe Based on Ingredients")
    recommend_expander1 = st.expander("Expand", expanded=True)
    with recommend_expander1:
        input_ingredients = st.multiselect(
            "Enter your ingredients list",
            words
        )
        
        btn_recommend1_clicked = st.button('Generate Recommendation')
        if btn_recommend1_clicked:
            input_str = ','.join(input_ingredients)
            # st.text('You selected:' + input_str)
            # st.text("Here is a list of recommended recipes...")
            st.spinner()
            with st.spinner(text="Loading..."):
                t = time()
                rec = get_recs(input_str, 10)
                st.text('Recommended recipes with the ingredients list:' + input_str + "\n")
                st.dataframe(rec)
                st.text('\nTime taken: {} mins'.format(round((time() - t) / 60, 2)))

if __name__ == '__main__':
    main()
    #print_sidebar_from_session()