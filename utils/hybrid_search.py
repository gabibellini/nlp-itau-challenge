from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import numpy as np
import pandas as pd
from pre_processing import preprocess_columns, funcs
from model import model

def hybrid_predict(
    user_input,
    user_uf,
    df_y,
    y_embeddings,
    k=5,
    levenshtein_threshold=0.75
):
    # Pré-processa user_input
    user_input_df = pd.DataFrame({'user_input': [user_input]})
    user_input_df = preprocess_columns(user_input_df, ['user_input'], funcs)
    user_input_proc = user_input_df['user_input'].iloc[0]

    # Pré-processa UF
    if user_uf:
        user_uf_df = pd.DataFrame({'uf': [user_uf]})
        user_uf_df = preprocess_columns(user_uf_df, ['uf'], funcs)
        user_uf_proc = user_uf_df['uf'].iloc[0]
    else:
        user_uf_proc = ''

    # Filtra por UF
    if user_uf_proc:
        mask = df_y['uf'].str.lower() == user_uf_proc
        filtered_df_y = df_y[mask].reset_index(drop=True)
        filtered_embeddings = y_embeddings[mask.values]
    else:
        filtered_df_y = df_y.reset_index(drop=True)
        filtered_embeddings = y_embeddings

    # Se nenhum candidato para o UF
    if len(filtered_df_y) == 0:
        return []

    # ---- Passo 1: Levenshtein ----
    levenshtein_scores = []
    max_lengths = []
    for target_text in filtered_df_y['y_text']:
        dist = levenshtein_distance(user_input_proc, target_text)
        max_len = max(len(user_input_proc), len(target_text))
        levenshtein_scores.append(dist)
        max_lengths.append(max_len)

    levenshtein_similarities = [
    1 - (dist / max_len) if max_len > 0 else 0
    for dist, max_len in zip(levenshtein_scores, max_lengths)
    ]

    # Verifica se tem candidatos bons o suficiente
    sorted_lev_indices = np.argsort([-sim for sim in levenshtein_similarities])
    top_lev_dist = levenshtein_similarities[sorted_lev_indices[0]]

    if top_lev_dist <= levenshtein_threshold:
        top_k_indices = sorted_lev_indices[:k]
        results = []
        for idx in top_k_indices:
            text = filtered_df_y.iloc[idx]['y_text']
            sim_score = levenshtein_similarities[idx]
            results.append((text, sim_score))  # menor distância = maior score
        return results


    # ---- Passo 2: Cosine Fallback ----
    user_embedding = model.encode([f"{user_input_proc} {user_uf_proc}"])[0]
    sims = cosine_similarity([user_embedding], filtered_embeddings)[0]
    top_k_indices = np.argsort(sims)[-k:][::-1]
    results = []
    for idx in top_k_indices:
        text = filtered_df_y.iloc[idx]['y_text']
        score = sims[idx]
        results.append((text, score))

    return results