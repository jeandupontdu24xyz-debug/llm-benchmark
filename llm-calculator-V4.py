import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration générale de la mini-app ---
st.set_page_config(page_title="Indice d'adaptabilité et de performance d'un LLM", page_icon="🧠", layout="wide")

st.title("🧠 Plateforme de calcul de l’indice d’adaptabilité et de performance d’un grand modèle de langage")

st.markdown("""
Cet outil interactif permet **d’évaluer et de comparer différents modèles de langage (LLM)**
selon leurs caractéristiques techniques, leur adaptabilité et compatibilité au fine-tuning / RAG, leur facilité de déploiement, 
dans le but d'obtenir un indice d'adaptabilité et de performance.
Il est également doté de **multiples fonctionnalités d'affichage** (graphique « radar », tableau comparatif, suivi colorimétrique de la performance de chaque facteur).
""")

# === Définiton des constantes de normalisation (à modifier en fonction des spécifités de votre système et de vos capacités matérielles) ===
P_MAX = 70
W_MAX = 80
V_MAX = 80
C_MAX = 128000
D_MAX = 10_000_000
E_MAX = 10

def normalize(value, max_value):
    return min(value / max_value, 1.0)

def normalize_format(fmt):
    mapping = {"FP32": 0.0, "FP16": 0.5, "8bit": 0.75, "4bit": 1.0, "NF4": 1.0}
    return mapping.get(fmt.upper(), 0.5)

def normalize_method(t):
    mapping = {
        "fine_tuning": 0.5,
        "lora": 0.7,
        "qlora": 0.8,
        "rag": 0.9,
        "rag_rlhf": 1.0
    }
    return mapping.get(t.lower(), 0.5)

# === INTERFACE UTILISATEUR ===
col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("Veuillez remplir les paramètres du modèle que vous souhaitez tester ⚙️")
    modele = st.text_input("Nom du modèle", "?")
    P = st.slider("Nombre de paramètres (en milliards)", 1.0, 70.0, 7.0)
    W = st.slider("Taille des poids (en Go)", 1.0, 80.0, 13.0)
    V = st.slider("VRAM requise (en Go)", 4.0, 80.0, 16.0)
    F = st.selectbox("Format des poids", ["FP32", "FP16", "8bit", "4bit", "NF4"], index=3)
    C = st.slider("Fenêtre de contexte maximale (tokens)", 1024, 128000, 32000, step=1024)
    D = st.number_input("Taille du dataset (tokens)", 10000, 10_000_000, 1_000_000)
    E = st.slider("Nombre d’époques", 1, 10, 3)
    T = st.selectbox("Méthode de fine-tuning", ["fine_tuning", "lora", "qlora", "rag", "rag_rlhf"], index=1)
    U = st.slider("Facilité de déploiement interne", 0.0, 1.0, 0.8)
    R = st.slider("Performance empirique perçue normalisée", 0.0, 1.0, 0.7)

with col2:
    st.header(" Pondérations des différents facteurs à l'aide de poids (automatiquement normalisés ⚖️)")
    w_P = st.slider("w_P (paramètres)", 0.0, 1.0, 0.15)
    w_W = st.slider("w_W (taille des poids)", 0.0, 1.0, 0.10)
    w_V = st.slider("w_V (VRAM)", 0.0, 1.0, 0.10)
    w_F = st.slider("w_F (format des poids)", 0.0, 1.0, 0.05)
    w_C = st.slider("w_C (contexte max)", 0.0, 1.0, 0.15)
    w_D = st.slider("w_D (dataset)", 0.0, 1.0, 0.10)
    w_E = st.slider("w_E (époques)", 0.0, 1.0, 0.05)
    w_T = st.slider("w_T (méthode de fine-tuning)", 0.0, 1.0, 0.20)
    w_U = st.slider("w_U (déploiement)", 0.0, 1.0, 0.10)
    w_R = st.slider("w_R (score  empirique)", 0.0, 1.0, 0.10)

# --- Normalisation automatique des poids ---
weights = [w_P, w_W, w_V, w_F, w_C, w_D, w_E, w_T, w_U, w_R]
total_weight = sum(weights)
if total_weight != 0:
    weights = [w / total_weight for w in weights]
w_P, w_W, w_V, w_F, w_C, w_D, w_E, w_T, w_U, w_R = weights

# === Calcul des scores normalisés ===
f_P = normalize(P, P_MAX)
f_W = normalize(W, W_MAX)
f_V = normalize(V, V_MAX)
f_F = normalize_format(F)
f_C = normalize(C, C_MAX)
f_D = normalize(D, D_MAX)
f_E = normalize(E, E_MAX)
f_T = normalize_method(T)
f_U = U
f_R = R

positifs = (w_P*f_P + w_C*f_C + w_D*f_D + w_E*f_E + w_T*f_T + w_U*f_U + w_R*f_R+ w_F*f_F)
negatifs = (w_W*f_W + w_V*f_V)
I = (positifs - negatifs)
I = max(0, min(I, 1))

# === Affichage du résultat ===
st.markdown("---")
st.subheader(f"Indice d’adaptabilité et de performance pour le LLM **{modele}** : {I:.3f}")

progress_color = "lime" if I > 0.9 else "limegreen" if I > 0.8 else "green" if I > 0.7 else "greenyellow" if I > 0.6 else "yellow" if I > 0.5 else "orange" if I > 0.4 else "darkorange" if I > 0.3 else "red"
st.progress(I)

# Production d'un retour automatique à l'utilisateur
if I > 0.8:
    st.success("🔝 **Ce LLM semble disposer d'une excellente capacité de spécialisation** — Il pourrait être très pertinent pour le besoin formulé.")
elif I > 0.6:
    st.info("✅ **Ce LLM semble disposer d'une capacité de spécialisation correcte, dans la moyenne** — Il peut convenir avec ajustement du fine-tuning ou du contexte.")
else:
    st.warning("⚠️ **Ce LLM semble disposer d'une capacité de spécialisation faible** — Il paraît peu adapté pour le cas d’usage.")

# === Tableau des valeurs calculées ===
data = {
    "Critère": ["Paramètres", "Taille", "VRAM", "Format", "Contexte", "Dataset", "Époques", "Fine-tuning", "Déploiement", "Benchmark"],
    "Score normalisé": [f_P, f_W, f_V, f_F, f_C, f_D, f_E, f_T, f_U, f_R],
    "Poids normalisé": [w_P, w_W, w_V, w_F, w_C, w_D, w_E, w_T, w_U, w_R],
}
df = pd.DataFrame(data)
st.dataframe(
    df.style.background_gradient(subset=["Score normalisé"], cmap="Greens")
             .format({"Score normalisé": "{:.2f}", "Poids normalisé": "{:.2f}"})
)

# === Affichage interactif du graphique « radar » ===
if st.button("📈 Afficher le graphique « radar »  du modèle"):
    categories = list(data["Critère"])
    values = list(data["Score normalisé"])
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color='green')
    ax.plot(angles, values, linewidth=2, color='green')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title(f"Profil de {modele}")
    st.pyplot(fig)

# === Outil de comparaison multi-modèles ===
st.markdown("### 🔄 Outil de comparaison multi-modèles")
if "models" not in st.session_state:
    st.session_state.models = {}

if st.button("Ajouter ce modèle à la liste des LLM testés"):
    st.session_state.models[modele] = I
    st.success(f"✅ {modele} ajouté à la liste de tests.")

if st.session_state.models:
    comp_df = pd.DataFrame(list(st.session_state.models.items()), columns=["Modèle", "Indice"]).sort_values("Indice", ascending=False)
    st.dataframe(comp_df)

    # Graphique de comparaison
    st.bar_chart(comp_df.set_index("Modèle"))

    # Export CSV
    csv = comp_df.to_csv(index=False)
    st.download_button("📥 Télécharger les résultats (.csv)", csv, "indices_llm.csv", "text/csv")
