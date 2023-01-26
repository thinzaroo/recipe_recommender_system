import pandas as pd
from numpy import dot
import random
from numpy.linalg import norm

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
    
    additionalColumns = ['ingredients_simple','url','nutr_values_per100g_energy','nutr_values_per100g_protein']
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
    # recipe_index = random.randint(1,51235)
    recipe_index = 31812
    df_recipes = pd.read_pickle('processed/df_recipes_with_embeddings.pkl')
    print('index:' + str(recipe_index))
    print("Recommend similar recipe for:", df_recipes.loc[recipe_index, "title"])

    res = list_Similar_dishes(df_recipes, recipe_index, "ingredients_embedding_w2v")
    print(res[0])

if __name__ == "__main__":
    main() 