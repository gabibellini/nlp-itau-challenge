import pandas as pd
from model.model import load_model, generate_embeddings
from utils.hybrid_search import hybrid_predict
import sys
import yaml
from utils.pre_processing import preprocess_columns, y_columns, uf_column, funcs


def main(
        test_path = 'data/train.parquet',

):
    confs = yaml.safe_load(open("data/confs.yaml"))
    text_target = confs["text_target"]

    user_input = pd.read_parquet(test_path)
    user_uf = user_input['uf']
    train = pd.read_parquet('data/train.parquet')
    uf, y_train = train['uf'], train[text_target]

    y_train = preprocess_columns(y_train, y_columns, funcs)
    uf_df = uf.to_frame()
    uf_df = preprocess_columns(uf_df, uf_column, funcs)

    y_train_sentences = y_train[y_columns].agg(' '.join, axis = 1)

    y_sentences = y_train_sentences.tolist()
    y_embeddings = generate_embeddings(load_model,y_sentences)
    df_y = y_train.copy()
    df_y['y_text'] = y_train_sentences
    df_y['uf'] = uf_df['uf']
    
    # args
    k = 5
    levenshtein_threshold = 0.75
    
    # predict
    results = hybrid_predict(
        user_input,
        user_uf,
        df_y=df_y,
        y_embeddings=y_embeddings,
        k=k,
        levenshtein_threshold=levenshtein_threshold
    )

    # results
    print(f"\nResultados para o input '{user_input}' (UF: {user_uf}):\n")
    for idx, (company_name, score) in enumerate(results, start=1):
        print(f"{idx}. {company_name} - Score: {score:.4f}")

if __name__ == "__main__":
    test_path = sys.argv[1]
    main(
        test_path
    )
