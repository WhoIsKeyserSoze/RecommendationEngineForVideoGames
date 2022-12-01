import pandas as pd


def generate_id_mappings(ids_list):
    userId_map = {new_id: old_id for new_id, old_id in enumerate(ids_list)}
    inverse_userId_map = {old_id: new_id for new_id, old_id in enumerate(ids_list)}
    return userId_map, inverse_userId_map


def recommend_gamesSVD(
    predictions_df,
    userId,
    inverse_user_mapping,
    games_df,
    original_ratings_df,
    num_recommendations=10,
):
    # Ordonner les prédictions pour l'utilisateur donné
    inverse_userId = inverse_user_mapping[userId]
    sorted_user_predictions = predictions_df.iloc[inverse_userId].sort_values(
        ascending=False
    )

    # Recupérer les notes déjà données par l'utilisateur avec les titres de jeux
    user_data = original_ratings_df[original_ratings_df.mUserId == (inverse_userId)]
    user_full = (
        user_data.merge(
            games_df, how="left", left_on="mGameId", right_on="mGameId"
        ).sort_values(["rating"], ascending=False)
    )[["mGameId", "Name_x", "rating"]]
    # print(user_full)
    # print('\n\nL\'utilisateur %d a déjà joué et indirectement noté %d jeux.' % (userId, user_full.shape[0]))

    # Recommander les jeux les mieux notés pas encore joué
    recommendations = (
        games_df[~games_df["mGameId"].isin(user_full["mGameId"])]
        .merge(
            pd.DataFrame(sorted_user_predictions).reset_index(),
            how="left",
            left_on="mGameId",
            right_on="mGameId",
        )
        .rename(columns={inverse_userId: "Predictions"})
        .sort_values("Predictions", ascending=False)
    )

    min_pred = recommendations['Predictions'].min()
    max_pred = recommendations['Predictions'].max()
    recommendations = recommendations.assign(
        Notes=lambda x: (5 * (x['Predictions'] - min_pred) / (max_pred - min_pred)))

    recommendations.drop(['Predictions'], axis=1, inplace=True)
    recommendations['Notes'] = recommendations['Notes'].round(1)
    recommendations = recommendations.iloc[:num_recommendations]
    return user_full, recommendations
