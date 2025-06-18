## Descrição

O modelo utiliza um mix de:

- Filtro por UF (localidade) - diminuir a quantidade de dados que precisa analisar
- Similaridade de Levenshtein - procura os candidatos mais próximos ao user input
- Similaridade de embeddings (Sentence Transformers + Cosine Similarity) - fall back

---

# Previsão

Para fazer a previsão, chamar a função main() com o path do dataset de teste.


## Estrutura de Pastas

projeto/
├── main_batch.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── model/
└── utils/

---

## Próximos passos
- Testar thresholds diferentes para Levenshtein
- Experimentar outros modelos de embeddings