import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- RIVOR Core Functions ---
def normalize_matrix(matrix, criteria_types, targets=None):
    norm_matrix = matrix.copy()
    for j in range(matrix.shape[1]):
        col = matrix.iloc[:, j]
        max_val, min_val = col.max(), col.min()

        if max_val == min_val:
            norm_matrix.iloc[:, j] = 1  # avoid division by zero
        elif criteria_types[j] == 'benefit':
            norm_matrix.iloc[:, j] = (col - min_val) / (max_val - min_val)
        elif criteria_types[j] == 'cost':
            norm_matrix.iloc[:, j] = (max_val - col) / (max_val - min_val)
        elif criteria_types[j] == 'target':
            Tj = targets[j] if targets is not None else (max_val + min_val) / 2
            norm_matrix.iloc[:, j] = 1 - abs(col - Tj) / (max_val - min_val)
    return norm_matrix

def calculate_weighted_matrix(norm_matrix, weights):
    return norm_matrix.multiply(weights, axis=1)

def calculate_S_R(weighted_matrix):
    S = weighted_matrix.sum(axis=1)
    R = weighted_matrix.max(axis=1)
    return S, R

def calculate_Q(S, R, v=0.5):
    S_min, S_max = S.min(), S.max()
    R_min, R_max = R.min(), R.max()
    Q = v * (S - S_min) / (S_max - S_min + 1e-10) + (1 - v) * (R - R_min) / (R_max - R_min + 1e-10)
    return Q

def rank_alternatives(Q):
    return Q.rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---
def run_app():
    st.title("RIVOR Decision Engine")
    st.write("Multi-Criteria Decision-Making System using RIVOR Method")

    st.header("Step 1: Upload Decision Matrix")
    uploaded_file = st.file_uploader("Upload a CSV file (alternatives as rows, criteria as columns)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, index_col=0)
        st.dataframe(df)

        st.header("Step 2: Enter Criteria Types and Weights")
        criteria = list(df.columns)
        criteria_types = []
        weights = []
        targets = []

        with st.form("criteria_form"):
            for col in criteria:
                col1, col2, col3 = st.columns(3)
                with col1:
                    ctype = st.selectbox(f"Type for {col}", ["benefit", "cost", "target"], key=col)
                with col2:
                    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"w_{col}")
                with col3:
                    target_val = st.number_input(f"Target (if target type)", value=0.0, step=0.1, key=f"t_{col}")
                criteria_types.append(ctype)
                weights.append(weight)
                targets.append(target_val)
            v = st.slider("Compromise parameter (v)", 0.0, 1.0, 0.5)
            submitted = st.form_submit_button("Run RIVOR Analysis")

        if submitted:
            weights = np.array(weights)
            if weights.sum() == 0:
                st.error("Total weight is zero. Please input valid weights.")
                return
            normalized_weights = weights / weights.sum()
            st.info(f"Normalized Weights: {dict(zip(criteria, np.round(normalized_weights, 3)))}")

            norm_matrix = normalize_matrix(df, criteria_types, targets)
            weighted_matrix = calculate_weighted_matrix(norm_matrix, normalized_weights)
            S, R = calculate_S_R(weighted_matrix)
            Q = calculate_Q(S, R, v)
            ranking = rank_alternatives(Q)

            results = pd.DataFrame({
                "Si (Utility)": S,
                "Ri (Regret)": R,
                "Qi (RIVOR Index)": Q,
                "Rank": ranking
            }, index=df.index)

            st.success("RIVOR Analysis Complete!")
            st.dataframe(results.style.highlight_min(axis=0, color='lightgreen').highlight_max(axis=0, color='lightblue'))

            # Bar chart
            st.subheader("📊 RIVOR Index Chart")
            fig, ax = plt.subplots()
            results["Qi (RIVOR Index)"].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Qi Value")
            ax.set_title("RIVOR Index by Alternative")
            st.pyplot(fig)

            # Download results
            st.subheader("⬇️ Download Results")
            csv = results.to_csv().encode('utf-8')
            st.download_button("Download CSV", csv, "rivor_results.csv", "text/csv")

if __name__ == "__main__":
    run_app()
