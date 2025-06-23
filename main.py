import os
import re
import pickle
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from functools import reduce
import operator

# Configuration de la page Streamlit
st.set_page_config(page_title="🎓 Analyse Scolaire", layout="centered")
st.title("🎓 Chatbot Scolaire - Analyse des Performances")

# Chargement des données avec cache
@st.cache_data(ttl=5184000)
def load_data():
    with open("df_finale.pkl", "rb") as f:
        df = pickle.load(f)
    return df

df_finale = load_data()

# Initialisation du modèle
llm = ChatOllama(model="gemma:2b", temperature=0.7)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "donnees"],
    template="""
Tu es un expert en analyse pédagogique,  conçue pour fournir des réponses précises, structurées et basées sur des données scolaires.

Voici des données sur les performances scolaires d'élèves d'une même classe. Chaque bloc correspond à un élève.

Ta tâche est :
### question concernant un élève :
**Pour un élève spécifique** (par identifiant_unique_eleve ou id_eleve) :
- Fournis ses notes (notes_matieres, moyenne_t1, moyenne_t2, moyenne_t3), son rang (rang_t1, rang_t2, rang_t3), et ses absences (type_presence, motif_absence).
- Analyse ses forces (matières avec hautes notes) et faiblesses (matières avec basses notes).
- Identifie les tendances (ex. matières difficiles, élèves performants,élève moyen, élève faible).
- Analyse ses résultats globaux et par matière.
- Compare sa performance à celle de sa classe.
- Repère ses points forts et ses difficultés.
- Fournis des suggestion et des conseils personnalisés pour son amélioration 

### question concernant une classe:
- donner l'effectif total de la classe (effectif_classe_t1) et par sexe (classe_effectif_1.0:masculin, classe_effectif_2.0:feminin)
- donné la moyenne générale de la classe (moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3).
- donnée le taux de reussite  de la classe (taux_reussite_classe_t1, taux_reussite_classe_t2, taux_reussite_classe_t3) et selon le sexe (taux_reussite_classe_1.0 : garcon,	taux_reussite_classe_2.0 : fille
)
- comparer les performances selon le sexe 
** identifie:
- Le meilleur et le plus faible élève selon la moyenne générale par trimestre, aussi la moyenne de la classe en se basant sur cette colonne (moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3)
- Utilise les statistiques pour les moyennes, maximums et effectifs pour une classe dans la colonne nom_salle_classe(CP1,CP2,CE1,CE2,CM1 et CM2) dans une école données
- Identifie les tendances 
- Repérer les matières les mieux et moins bien réussies
- Indiquer s'il existe des cas exceptionnels (très bons ou très faibles)
- Donne un aperçu des écarts de performance.
- Propose des suggestions et des pistes pédagogiques concrètes pour renforcer les acquis ou combler les lacunes.

### question concernan une école:
**Dresse un bilan *par classe* :
- Nombre de classe (nb_classes)
- effectif de l'école (effectif_total_ecole)
- Moyenne générale de chaque classe (moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3).
** Intègre aussi :
- Effectif global des enseignants et selon le sexe(1:masculin, 2:feminin)
- Effectifs global des élèves et selon le sexe(1:masculin, 2:feminin)
- Présence de cantine
- Présence de latrines/toilettes/WC
- Présence de fontaine/pompe/eau potable
- Présence d'électricité
- Milieu: urbain ou rural
- Matériels didactiques
- Performances des élèves de façon globale et par sexe(moyenne par trimestre, matières réussies et moins réussies)
- Assiduité (absences, présences, abandons) global et par sexe(1:masculin, 2:feminin)
-  Les cas de *violence ou de victimisation* s'ils sont signalés.
- Les caractéristiques spécifiques de l'école (environnement, effectif, encadrement, etc.).
- Suggère des recommandations réalistes pour améliorer la qualité de l'enseignement dans l'établissement.

###Si la question concerne une CEB ou une commune
**Présente une *analyse comparative entre écoles* :
- Nombre d'écoles
- Nombre d'enseignants et par sexe
- Nombre élèves et par sexe
- Ratio élèves/Enseignants
- Proportion d'écoles sans cantine
- Proportion d'écoles sans latrines
- Proportion d'écoles sans électricité 
- Nombre de PDI en prenant comme variable le statut_eleve(2:PDI)
- Nombre d'élèves avec handicap
- Performances des élèves de façon globale et par sexe(moyenne par trimestre, matières réussies et moins réussies)
- Assiduité (absences, présences, abandons) global et par sexe
- Performances globales (par classe et par école).
- Classement ou hiérarchisation des écoles si pertinent.
- Forces et faiblesses communes ou spécifiques.
- Signalement des situations problématiques (violences, inégalités, déséquilibres).
- Propose des recommandations *à l'échelle territoriale* (CEB ou commune) pour renforcer l'apprentissage et réduire les disparités.

###Objectif final 
**Fournir une *analyse claire, structurée et compréhensible*, avec :
- Des *constats basés sur les données*.
- Des *conclusions pédagogiques* pertinentes.
- Des *recommandations pratiques* pour améliorer les performances à tous les niveaux analysés.

**Ne jamais inventer de données**. Si les données sont manquantes, indique-le clairement.


Question : {question}

Données :
{donnees}

Fais une réponse claire et structurée.
"""
)

# Fonction pour détecter un filtre dans la question
def extraire_filtre(question, valeurs_connues):
    for val in valeurs_connues:
        if val and str(val).lower() in question.lower():
            return val
    return None

def get_response_from_dataframe(question, df):
    from functools import reduce
    import operator
    reponses = []

    question_lower = question.lower()

    # Recherche des filtres possibles
    id_eleve = extraire_filtre(question_lower, df['id_eleve'].astype(str).unique())
    identifiant_unique = extraire_filtre(question_lower, df['identifiant_unique_eleve'].astype(str).unique())
    id_classe = extraire_filtre(question_lower, df['id_classe'].astype(str).unique())
    code_classe = extraire_filtre(question_lower, df['code_classe'].astype(str).unique())
    nom_classe = extraire_filtre(question_lower, df['nom_classe'].astype(str).unique())
    nom_ecole = extraire_filtre(question_lower, df['nom_ecole'].astype(str).unique())
    code_ecole = extraire_filtre(question_lower, df['code_ecole'].astype(str).unique())
    ceb = extraire_filtre(question_lower, df['ceb_ecole'].astype(str).unique())
    commune = extraire_filtre(question_lower, df['commune_ecole'].astype(str).unique())
    id_ecole = extraire_filtre(question_lower, df['id_ecole'].astype(str).unique())

    # 🔍 Élève
    if id_eleve or identifiant_unique:
        ident = id_eleve or identifiant_unique
        ligne = df[(df['id_eleve'].astype(str) == ident) | (df['identifiant_unique_eleve'].astype(str) == ident)]
        if not ligne.empty:
            ligne = ligne.iloc[0]
            donnees_texte = "\n".join([f"{col} : {ligne[col]}" for col in df.columns if col in ligne])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

    # 🔍 Classe / école
    filtres = []
    if nom_ecole: filtres.append(df['nom_ecole'].str.lower() == nom_ecole.lower())
    if code_ecole: filtres.append(df['code_ecole'].astype(str) == str(code_ecole))
    if ceb: filtres.append(df['ceb_ecole'].astype(str) == str(ceb))
    if commune: filtres.append(df['commune_ecole'].astype(str) == str(commune))
    if code_classe: filtres.append(df['code_classe'].astype(str) == str(code_classe))
    if nom_classe: filtres.append(df['nom_classe'].str.lower() == nom_classe.lower())
    if id_classe: filtres.append(df['id_classe'].astype(str) == str(id_classe))
    if id_ecole: filtres.append(df['id_ecole'].astype(str) == str(id_ecole))

    if filtres:
        condition = reduce(operator.and_, filtres)
        df_filtre = df[condition]
        if df_filtre.empty:
            return "Aucune donnée trouvée avec les critères spécifiés."

        # Fixer automatiquement le nombre d’élèves dans la classe /On évite d’afficher tous les élèves si ce n’est pas explicitement demandé.
        nb_eleves = df_filtre.shape[0]

        # 🎯 Analyse par classe
        if "classe" in question_lower or "classes" in question_lower:
            classes = df_filtre['nom_classe'].unique()
            for classe in classes:
                df_classe = df_filtre[df_filtre['nom_classe'] == classe]
                resume = {col: df_classe[col].mean() for col in df_classe.columns if df_classe[col].dtype != 'object'}
                donnees_texte = f"Classe : {classe}\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
                prompt = prompt_template.format(question=question, donnees=donnees_texte)
                resultat = llm.invoke(prompt)
                if hasattr(resultat, 'content'):
                    resultat = resultat.content
                reponses.append(f"Classe {classe} :\n{resultat}")
            return "\n\n---\n\n".join(reponses)

        # 🎯 Analyse globale de l’école
        elif "école" in question_lower or "ecole" in question_lower or "établissement" in question_lower:
            resume = {col: df_filtre[col].mean() for col in df_filtre.columns if df_filtre[col].dtype != 'object'}
            donnees_texte = f"Ecole : {df_filtre['nom_ecole'].iloc[0]}\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

        # 🎯 Si CEB ou commune
        elif "ceb" in question_lower or "commune" in question_lower:
            resume = df_filtre.groupby("nom_ecole").mean(numeric_only=True)
            donnees_texte = resume.round(2).to_string()
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

        # 🔄 Sinon (traitement classe sans mention explicite) : résumé sans nommer les élèves
        resume = {col: df_filtre[col].mean() for col in df_filtre.columns if df_filtre[col].dtype != 'object'}
        donnees_texte = "Résumé global :\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
        prompt = prompt_template.format(question=question, donnees=donnees_texte)
        resultat = llm.invoke(prompt)
        return resultat.content if hasattr(resultat, 'content') else resultat

    return "Aucun filtre détecté dans la question. Veuillez spécifier un élève, une classe ou une école."

# Initialisation de l'historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Affichage de l'historique dans la barre latérale (à gauche)
with st.sidebar:
    st.header("📜 Historique")
    
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            role = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"{role} {msg['content']}")
    else:
        st.info("Aucun échange pour le moment.")
    
    if st.button("🧹 Effacer l'historique"):
        st.session_state.chat_history = []
        st.success("Historique effacé.")

# Titre principal
#st.title("Chatbot Scolarité")

# Saisie utilisateur
user_input = st.chat_input("Pose ta question")

if user_input:
    # Générer la réponse (fonction personnalisée)
    response = get_response_from_dataframe(user_input, df_finale)

    # Sauvegarde dans l'historique
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.write(user_input)

    # Affichage de la réponse
    with st.chat_message("assistant"):
        st.write(response)


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
