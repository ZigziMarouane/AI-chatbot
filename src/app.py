# Configuration de la page (doit être le premier appel Streamlit)
import streamlit as st
st.set_page_config(
    page_title="Assistant",
    page_icon=":school:",
    layout="wide"
)

# Importation des bibliothèques nécessaires
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_debug
import logging
from os import getenv

# Configuration avancée du système de journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()
set_debug(False)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# Initialisation de l'état de session
if 'db' not in st.session_state:
    try:
        db_uri = f"mysql+mysqlconnector://{getenv('DB_USER', 'root')}:{getenv('DB_PASS', '')}@{getenv('DB_HOST', 'localhost')}:{getenv('DB_PORT', '3306')}/{getenv('DB_NAME', 'chatbotbd')}"
        st.session_state.db = SQLDatabase.from_uri(db_uri)
        logger.info("Connexion à la base de données établie avec succès")
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {str(e)}")
        st.session_state.db = None
if 'llm' not in st.session_state:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0
        )
        logger.info("Modèle LLM initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur d'initialisation du LLM: {str(e)}")
        st.session_state.llm = None

# Configuration de la base de données
DB_CONFIG = {
    'user': getenv('DB_USER', 'root'),
    'password': getenv('DB_PASS', ''),
    'host': getenv('DB_HOST', 'localhost'),
    'port': getenv('DB_PORT', '3306'),
    'database': getenv('DB_NAME', 'chatbotbd')
}

# Modification de la fonction init_ressources
@st.cache_resource
def init_ressources():
    """Vérification et réinitialisation des ressources si nécessaire"""
    if st.session_state.db is None:
        try:
            db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            st.session_state.db = SQLDatabase.from_uri(db_uri)
            logger.info("Reconnexion à la base de données réussie")
        except Exception as e:
            logger.error(f"Échec de reconnexion à la base de données: {str(e)}")
            
    if st.session_state.llm is None:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0
            )
            logger.info("Réinitialisation du modèle LLM réussie")
        except Exception as e:
            logger.error(f"Échec de réinitialisation du LLM: {str(e)}")

def get_sql_chain(db):
    # Définition du template pour la génération des requêtes SQL
    template = """
    You are an assistant at a vocational school to help trainees apply. You are interacting with a user who is asking you questions about the school's database.
    Based on the table schema below, write a mySQL query that would answer the user's question. Take the conversation history into account.
    
    IMPORTANT:
    1. ALWAYS check ALL relevant tables
    2. Use JOIN operations when needed to get complete information
    3. Never limit results unless specifically asked
    4. when a use gives their bac option, verify its admissibility from the admissibilite table.
    5. bac info is branches_bac table.
    6. bac to formation admissibility is in the admissibilite table.
    7. second year options are in the option_2eme_annee table with id_formation foreign key to the table Formations id.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the raw mySQL query without any markdown formatting, backticks, or prefixes. The query should be directly executable.
    Do not include any additional text or explanations.
    Do not include any comments in the mySQL query.
    For example:
    Question: les options disponible pour developpement digital?
    SQL Query: SELECT o.nom , o.description FROM option_2eme_annee o 
    JOIN filieres_est f ON o.id_formation = f.id 
    WHERE f.nom LIKE '%Développement Digital%';
    Question: which filières are available?
    SQL Query: SELECT nom FROM filieres_est;
    Question: what are the requirements for GI?
    SQL Query: SELECT bac_option, coefficient FROM conditions WHERE filiere = 'GI';
    
    Your turn:
    
    Question: {question}
    SQL Query:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        # Initialisation du modèle de langage Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    except Exception as e:
        logger.error(f"Erreur d'initialisation du LLM: {str(e)}")
        return None
    
    def get_schema(_):
        try:
            # Récupération du schéma de la base de données
            return db.get_table_info()
        except Exception as e:
            logger.error(f"Erreur de récupération du schéma: {str(e)}")
            return ""
    
    # Configuration de la chaîne de traitement
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    try:
        # Initialisation de la chaîne SQL
        sql_chain = get_sql_chain(db)
        if sql_chain is None:
            logger.error("Échec de l'initialisation de la chaîne SQL")
            return "Je m'excuse, je ne peux pas répondre à votre question pour le moment."
        
        # Configuration du template de réponse
        template = """
        You are an assistant at ISTA NTIC (Institut Spécialisés dans les Métiers de l'Offshoring et les Nouvelles Technologies de l'Information). 
        You have access to the database.
        consider the entire database when answering the user's question. 
        split modules by their type.
        always make your answers easy to look at.
        for the conditions of admission, specify that diplome is either a bac or diploma(systeme de passerelle).
        when you display the conditions of admission, ask if they want to know about the admission papers.
        for modules, always put the name of the module in the first column and the type in the second column.
        
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}

        Write your response in the language the user asked with (English,French or Arabic).
        Make your response natural and informative."""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        # Configuration de la chaîne de traitement principale
        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Génération de la réponse
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
    except Exception as e:
        logger.error(f"Erreur dans get_response: {str(e)}")
        return "Je m'excuse, je ne peux pas répondre à votre question pour le moment."

# Initialisation de l'historique des conversations
if "chat_history" not in st.session_state:
    welcome_message = """
    Bonjour ! Je suis un assistant virtuel, Comment puis-je vous assister aujourd'hui ?
    """
    st.session_state.chat_history = [
        AIMessage(content=welcome_message),
    ]

def main():
    """Fonction principale de l'application"""
    try:
        # Initialisation des ressources
        init_ressources()
        st.markdown("<h1 style='color: #4CAF50;'>Assistant chatbot</h1>", unsafe_allow_html=True)

        # Affichage de l'historique des messages
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        # Gestion des entrées utilisateur
        user_query = st.chat_input("message...")
        if user_query is not None and user_query.strip() != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            with st.chat_message("AI"):
                try:
                    # Génération et affichage de la réponse
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
                except Exception as e:
                    # Gestion des erreurs
                    logger.error(f"Erreur de traitement de la requête: {str(e)}")
                    error_response = "Je m'excuse, je ne peux pas répondre à votre question pour le moment."
                    st.markdown(error_response)
                    st.session_state.chat_history.append(AIMessage(content=error_response))
    except Exception as e:
        # Gestion des erreurs de l'interface utilisateur
        logger.error(f"Erreur de l'application: {str(e)}")
        st.error("Une erreur s'est produite. Veuillez réessayer plus tard.")

if __name__ == "__main__":
    main()