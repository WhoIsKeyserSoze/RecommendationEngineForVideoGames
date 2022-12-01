import uvicorn
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds  # SVD
from scipy.linalg import sqrtm

from api.utils import *

app = FastAPI()

@app.get("/model_categorie/")
def predict(categorie: str = Query(default=None)):
    df_genre = pd.read_csv('Datasets/df_genres.csv')
    mydict = {"categorie":categorie}
    genre = mydict["categorie"]
    print(genre)
    if genre not in df_genre.columns:
        print("Erreur : Catégorie invalide")
        return {"erreur" : "Erreur : Catégorie invalide"}
    df = df_genre.loc[df_genre[genre] == 1]
    df = df.sort_values(['rating'], ascending=False)
    games_list = []
    for i in range(min(len(df), 10)):
        games_list.append(df["Name"].iat[i])
    games_list = np.array(games_list)
    return {"recomendations": games_list.tolist()}


@app.get("/model_game_names/")
def predict(games_list: list[str] = Query(default=None)):
    games_dict = {"games_list": games_list}
    gamesList = games_dict["games_list"]
    if gamesList[0] == "" and len(games_list) == 1:
        print("Erreur : Nom de jeu invalide")
        return {"erreur" : "Erreur : Nom de jeu invalide"}

    latent_dimension = 30
    df_steam = pd.read_csv("Datasets/steam-200k.csv")
    df_steam.columns = ["UserID", "Name", "Usage", "HoursPlayed", "0"]
    df_steam = df_steam.drop(columns="0")
    for games in gamesList:
        new_row = {
            "UserID": 128479999,
            "Name": games,
            "Usage": "Play",
            "HoursPlayed": 999,
        }
        df_steam = df_steam.append(new_row, ignore_index=True)

    print(df_steam.loc[df_steam["UserID"] == 128479999])

    df_games = pd.read_csv("Datasets/games.csv")
    df_games.drop(
        [
            "main_story",
            "main_plus_extras",
            "completionist",
            "coop",
            "versus",
            "type",
            "developers",
            "release_na",
            "release_eu",
            "release_jp",
            "platforms",
            "publishers",
        ],
        axis=1,
        inplace=True,
    )
    df_games["all_styles"].fillna((df_games["all_styles"].mean()), inplace=True)

    df_merge = pd.merge(
        df_steam,
        df_games,
        left_on="Name",
        right_on="title",
        indicator=True,
        how="inner",
    )
    # On garde uniquement les jeux qui ont été entièrement joint
    # df_merge = df_merge[df_merge._merge != 'left_only']
    # On supprime la colonne _merge pour rendre le dataset plus lisible
    df_merge = df_merge.drop(columns="_merge")
    df_merged_games = df_merge[df_merge.Usage != "purchase"]

    condition = [
        df_merged_games["HoursPlayed"] >= (0.8 * df_merged_games["all_styles"]),
        (df_merged_games["HoursPlayed"] >= 0.6 * df_merged_games["all_styles"])
        & (df_merged_games["HoursPlayed"] < 0.8 * df_merged_games["all_styles"]),
        (df_merged_games["HoursPlayed"] >= 0.4 * df_merged_games["all_styles"])
        & (df_merged_games["HoursPlayed"] < 0.6 * df_merged_games["all_styles"]),
        (df_merged_games["HoursPlayed"] >= 0.2 * df_merged_games["all_styles"])
        & (df_merged_games["HoursPlayed"] < 0.4 * df_merged_games["all_styles"]),
        df_merged_games["HoursPlayed"] >= 0,
    ]
    values = [5, 4, 3, 2, 1]
    df_merged_games["rating"] = np.select(condition, values)

    userId_map, inverse_userId_map = generate_id_mappings(
        df_merged_games.UserID.unique()
    )
    gameId_map, inverse_gameld_map = generate_id_mappings(df_merged_games.id.unique())

    df_merged_games["mUserId"] = df_merged_games["UserID"].map(inverse_userId_map)
    df_merged_games["mGameId"] = df_merged_games["id"].map(inverse_gameld_map)

    df_exploit_games = df_merged_games[["mGameId", "Name", "genres"]].copy()
    df_exploit_games.to_csv("df_exploit_games.csv", index=False)

    df_exploit_CF = df_merged_games[["mUserId", "mGameId", "Name", "rating"]].copy()
    df_exploit_CF.to_csv("df_exploit_CF.csv", index=False)

    df_exploit_games = df_exploit_games.drop_duplicates()

    R_df = df_exploit_CF.pivot_table(
        index="mUserId", columns="mGameId", values="rating"
    ).fillna(0)
    R = R_df.values

    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k=latent_dimension)

    sigma = np.diag(sigma)  # transforme en matrice (50, 50)
    s_root = sqrtm(sigma)  # racine carré

    Usk = np.dot(U, s_root)
    skV = np.dot(s_root, Vt)

    predicted_rating = np.dot(Usk, skV)
    predicted_rating = predicted_rating + user_ratings_mean.reshape(-1, 1)

    prediction_df = pd.DataFrame(predicted_rating, columns=R_df.columns)

    already_rated, predictions = recommend_gamesSVD(
        prediction_df,
        128479999,
        inverse_userId_map,
        df_exploit_games,
        df_exploit_CF,
        10,
    )

    games_list = []
    for title in predictions["Name"]:
        games_list.append(title)

    notes_list = []
    for title in predictions["Notes"]:
        notes_list.append(title)

    return {"predictions": games_list, "notes" : notes_list}


if __name__ == "__main__":
    uvicorn.run("main:app")
