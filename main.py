import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- RIVOR Core Functions ---

def normalize_matrix(matrix, criteria_types, targets=None):
    norm_matrix = matrix.copy()
    for j in range(matrix.shape[1]):
        col = matrix.iloc[:, j]
        f_max = col.max()
        f_min = col.min()

        if f_max == f_min:
            norm_matrix.iloc[:, j] = 0
            continue

        if criteria_types[j] == 'benefit':
            norm_matrix.iloc[:, j] = (f_max - col) / (f_max - f_min)
        elif criteria_types[j] == 'cost':
            norm_matrix.iloc[:, j] = (col - f_min) / (f_max - f_min)
        elif criteria_types[j] == 'target':
            Tj = targets[j] if targets is not None else (f_max + f_min) / 2
            denom = max(abs(f_max - Tj), abs(f_min - Tj))
            norm_matrix.iloc[:, j] = abs(col - Tj) / denom
    return norm_matrix

def calculate_weighted_matrix(norm_matrix, weights):
    return norm_matrix.multiply(weights, axis=1)

def calculate_S_R(weighted_matrix):
    S = weighted_matrix.sum(axis=1)
    R = weighted_matrix.max(axis=1)
    return S, R

def calculate_Q(S, R, v=0.5):
    S_star = S.min()
    S_minus = S.max()
    R_star = R.min()
    R_minus = R.max()
    S_range = S_minus - S_star + 1e-10
    R_range = R_minus - R_star + 1e-10
    Q = v * (S_minus - S) / S_range + (1 - v) * (R_minus - R) / R_range
    return Q

def rank_alternatives(Q):
    return Q.rank(ascending=False, method='min').astype(int)

# --- Streamlit App ---

def run_app():
    st.title("RIVOR Decision Engine")
    st.write("Multi-Criteria Decision-Making System using the RIVOR Method")

    st.header("Step 1: Upload Decision Matrix")
    uploaded_file = st.file_uploader("Upload a CSV file (alternatives as rows, criteria as columns)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, index_col=0)
        st.dataframe(df)

        st.header("Step 2: Enter Criteria Types and Weights")
        criteria = list(df.columns)

        with st.form("criteria_form"):
            criteria_types = []
            weights = []
            targets = []

            num_criteria = len(criteria)
            cols_per_row = 4 if num_criteria <= 20 else 2

            for i, col_name in enumerate(criteria):
                if i % cols_per_row == 0:
                    columns = st.columns(cols_per_row)
                col = columns[i % cols_per_row]

                with col.expander(f"⚙️ {col_name}", expanded=False):
                    ctype = st.selectbox("Type", ["benefit", "cost", "target"], key=f"type_{col_name}")
                    weight = st.number_input("Weight", 0.0, 1.0, 1.0, 0.1, key=f"w_{col_name}")
                    target_val = st.number_input("Target (if target type)", 0.0, step=0.1, key=f"t_{col_name}")

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

            # Step 1: Normalization
            norm_matrix = normalize_matrix(df, criteria_types, targets)
            st.subheader("🔹 Step 1: Normalized Decision Matrix")
            st.dataframe(norm_matrix.style.format("{:.4f}"))

            # Step 2: Weighted Normalized Matrix
            weighted_matrix = calculate_weighted_matrix(norm_matrix, normalized_weights)
            st.subheader("🔹 Step 2: Weighted Normalized Matrix")
            st.dataframe(weighted_matrix.style.format("{:.4f}"))

            # Step 3: Utility and Regret
            S, R = calculate_S_R(weighted_matrix)
            st.subheader("🔹 Step 3: Utility (Si) and Regret (Ri)")
            st.dataframe(pd.DataFrame({"Si (Utility)": S, "Ri (Regret)": R}).style.format("{:.4f}"))

            # Step 4: Qi and Ranking
            Q = calculate_Q(S, R, v)
            ranking = rank_alternatives(Q)
            results = pd.DataFrame({
                "Si (Utility)": S,
                "Ri (Regret)": R,
                "Qi (RIVOR Index)": Q,
                "Rank": ranking
            }, index=df.index)

            st.subheader("🔹 Step 4: RIVOR Index (Qi) and Final Ranking")
            st.dataframe(results.style.highlight_min("Rank", color="lightgreen").highlight_max("Qi (RIVOR Index)", color="lightblue"))

            # Step 5: Bar Chart
            st.subheader("📊 Step 5: RIVOR Index Chart")
            fig, ax = plt.subplots()
            results["Qi (RIVOR Index)"].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Qi Value")
            ax.set_title("RIVOR Index by Alternative")
            st.pyplot(fig)

            # Step 6: Download
            st.subheader("⬇️ Step 6: Download Results")
            csv = results.to_csv().encode('utf-8')
            st.download_button("Download CSV", csv, "rivor_results.csv", "text/csv")

if __name__ == "__main__":
    run_app()
