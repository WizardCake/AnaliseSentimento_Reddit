import praw
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Garantir a presença do recurso vader para análise de sentimento
nltk.download('vader_lexicon')

# Função para coletar comentários
def coletar_comentarios(subreddit, palavra_chave, quantidade_posts, quantidade_comentarios):
    """
    Busca comentários de posts de um subreddit específico contendo a palavra-chave.

    Args:
        subreddit (str): O nome do subreddit.
        palavra_chave (str): Palavra-chave para filtrar os posts.
        quantidade_posts (int): Quantidade de posts a serem buscados.
        quantidade_comentarios (int): Quantidade de comentários por post.

    Returns:
        list: Lista de comentários.
    """

    reddit = praw.Reddit(
        client_id='', #[CLIENT-ID-HERE]
        client_secret='', #[CLIENT-SECRET-HERE]
        user_agent="Analise_Sentimento/0.0.1"
    )

    subreddit_filtrado = reddit.subreddit(subreddit).search(palavra_chave, limit=quantidade_posts)

    comentarios_selecionados = []

    for post in subreddit_filtrado:
        post = reddit.submission(id=post.id)
        comentarios = post.comments
        for comentario in comentarios[:quantidade_comentarios]:
            comentarios_selecionados.append(comentario.body)

    return comentarios_selecionados

# Função para tratar os dados e analisa-los
def preparar_e_analisar_sentimentos(comentarios):
    """
    Prepara os dados dos comentários para análise de sentimento e realiza a análise.

    Args:
        comentarios (list): Lista de textos de comentários.

    Returns:
        DataFrame: DataFrame com os resultados da análise de sentimento.
    """
    df = pd.DataFrame(comentarios, columns=['text'])
    if df.empty:
        return df

    sia = SentimentIntensityAnalyzer()
    df['Sentimento'] = df['text'].apply(lambda texto: sia.polarity_scores(texto))
    df['Classificação'] = df['Sentimento'].apply(
        lambda score: 'positivo' if score['compound'] > 0 else 'negativo' if score['compound'] < 0 else 'neutro')

    return df[['text', 'Classificação']]


# Configuração dos parâmetros de busca
subreddit = 'technology'
palavra_chave = "chatbot"
quantidade_posts = 500
quantidade_comentarios = 50

# Execução da coleta e análise
comentarios = coletar_comentarios(subreddit, palavra_chave, quantidade_posts, quantidade_comentarios)

if comentarios:
    df_analisado = preparar_e_analisar_sentimentos(comentarios)
    # Visualização dos resultados
    sns.countplot(x='Classificação', data=df_analisado)
    plt.title('Análise de Sentimento dos Comentários sobre Chatbots no Reddit')
    plt.xlabel('Sentimento')
    plt.ylabel('Quantidade')
    plt.show()
else:
    print("Falha ao coletar comentários. Verifique os parâmetros e a conexão à internet.")