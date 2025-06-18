from utils.hybrid_search import hybrid_predict

def evaluate_precision_at_k_batch_hybrid(
    user_inputs,
    user_ufs,
    true_targets,
    df_y_path,
    y_embeddings_path,
    k=5,
    batch_size=100
):
    total = len(user_inputs)
    correct = 0

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_inputs = user_inputs[start_idx:end_idx]
        batch_ufs = user_ufs[start_idx:end_idx]
        batch_targets = true_targets[start_idx:end_idx]

        for idx in range(len(batch_inputs)):
            preds = hybrid_predict(
                user_input=batch_inputs[idx],
                user_uf=batch_ufs[idx],
                df_y_path=df_y_path,
                y_embeddings_path=y_embeddings_path,
                k=k
            )

            pred_texts = [p[0] for p in preds]
            if batch_targets[idx] in pred_texts:
                correct += 1

        print(f"Processed batch {start_idx}-{end_idx} | Precision so far: {correct / (end_idx):.4f}")

    final_precision = correct / total
    return final_precision
