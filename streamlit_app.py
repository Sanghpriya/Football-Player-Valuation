import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# File paths
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')

# Pointing to the dataset that still retains string columns (Name, Team) 
# but has the correct row count matching the engineered features.
CLEANED_CSV = DATA_DIR / 'data_after_feature_engineering.csv'
FEATURE_CSV = DATA_DIR / 'final_feature_engineering.csv'

MODEL_FILES = {
    'Gradient Boosting': MODEL_DIR / 'gb_model.pkl',
    'Stacked': MODEL_DIR / 'stack_model.pkl',
}

FEATURES = [
    'Age',
    'Potential',
    'foot',
    'Wage',
    'Height_cm',
    'Weight_kg',
    'Acceleration',
    'Sprint speed',
    'Agility',
    'Balance',
    'Stamina',
    'Strength',
    'International reputation',
    'Is_On_Loan',
    'Is_Free_Agent',
    'Contract_Years_Left',
    'Team_encoded',
    'Forward Score',
    'Midfielder Score',
    'Defender Score',
    'Goalkeeper Score',
    'Position Category_Defender',
    'Position Category_Forward',
    'Position Category_Goalkeeper',
    'Position Category_Midfielder',
]

# Order: [Defender, Forward, Goalkeeper, Midfielder]
POSITION_TO_ONE_HOT = {
    'Defender': [1, 0, 0, 0],
    'Forward': [0, 1, 0, 0],
    'Goalkeeper': [0, 0, 1, 0],
    'Midfielder': [0, 0, 0, 1],
}


def format_currency(amount: float) -> str:
    """Format large numbers into readable EUR values."""
    if amount >= 1e6:
        return f'€{amount / 1e6:.2f}M'
    if amount >= 1e3:
        return f'€{amount / 1e3:.2f}K'
    return f'€{amount:,.2f}'


@st.cache_data
def load_data():
    df_raw = pd.read_csv(CLEANED_CSV)
    df_processed = pd.read_csv(FEATURE_CSV)

    if len(df_raw) != len(df_processed):
        raise ValueError(f'Raw ({len(df_raw)}) and processed ({len(df_processed)}) feature files must have the same number of rows.')

    df_raw = df_raw.reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)
    df_raw['Team_encoded'] = df_processed['Team_encoded']

    for col in [
        'Position Category_Defender',
        'Position Category_Forward',
        'Position Category_Goalkeeper',
        'Position Category_Midfielder',
    ]:
        df_raw[col] = df_processed[col]

    # Load mapping cleanly from your JSON file
    try:
        with open('team_target_encoding.json', 'r') as f:
            team_mapping = json.load(f)
    except FileNotFoundError:
        st.warning("team_target_encoding.json not found, attempting to extract from CSV...")
        # Fallback to the old method if the JSON is missing
        team_mapping = (
            df_raw[['Team', 'Team_encoded']]
            .drop_duplicates(subset=['Team'])
            .set_index('Team')['Team_encoded']
            .to_dict()
        )

    return df_raw, df_processed, team_mapping


@st.cache_data
def load_model(model_name: str):
    model_path = MODEL_FILES.get(model_name)
    if model_path is None or not model_path.exists():
        st.error(f'Model file not found: {model_path}')
        return None
    return joblib.load(model_path)


def preprocess_raw_input(raw: dict, team_mapping: dict) -> pd.DataFrame:
    team = raw['Team']
    team_encoded = team_mapping.get(team, np.nan)
    if pd.isna(team_encoded):
        team_encoded = np.mean(list(team_mapping.values()))

    wage = float(raw['Wage'])
    wage_transformed = np.log1p(max(0.0, wage))

    data = {
        'Age': raw['Age'],
        'Potential': raw['Potential'],
        'foot': 1 if raw['foot'] == 'Right' else 0,
        'Wage': wage_transformed,
        'Height_cm': raw['Height_cm'],
        'Weight_kg': raw['Weight_kg'],
        'Acceleration': raw['Acceleration'],
        'Sprint speed': raw['Sprint speed'],
        'Agility': raw['Agility'],
        'Balance': raw['Balance'],
        'Stamina': raw['Stamina'],
        'Strength': raw['Strength'],
        'International reputation': raw['International reputation'],
        'Is_On_Loan': raw['Is_On_Loan'],
        'Is_Free_Agent': raw['Is_Free_Agent'],
        'Contract_Years_Left': raw['Contract_Years_Left'],
        'Team_encoded': team_encoded,
        'Forward Score': raw['Forward Score'],
        'Midfielder Score': raw['Midfielder Score'],
        'Defender Score': raw['Defender Score'],
        'Goalkeeper Score': raw['Goalkeeper Score'],
        'Position Category_Defender': raw['Position Category_Defender'],
        'Position Category_Forward': raw['Position Category_Forward'],
        'Position Category_Goalkeeper': raw['Position Category_Goalkeeper'],
        'Position Category_Midfielder': raw['Position Category_Midfielder'],
    }
    return pd.DataFrame([data], columns=FEATURES)


def build_processed_input() -> pd.DataFrame:
    st.header('Processed feature input')

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=15, max_value=45, value=22)
        potential = st.number_input('Potential', min_value=40, max_value=99, value=80)
        foot = st.selectbox('Preferred foot', options=[('Right', 1), ('Left', 0)], format_func=lambda x: x[0])[1]
        wage = st.number_input('Wage (log scale)', min_value=0.0, max_value=15.0, value=10.0, step=0.1)
        height = st.number_input('Height (cm)', min_value=150, max_value=210, value=180)
        weight = st.number_input('Weight (kg)', min_value=55, max_value=110, value=75)
        acceleration = st.number_input('Acceleration', min_value=1, max_value=99, value=75)
        sprint_speed = st.number_input('Sprint speed', min_value=1, max_value=99, value=75)
        agility = st.number_input('Agility', min_value=1, max_value=99, value=75)
        balance = st.number_input('Balance', min_value=1, max_value=99, value=70)

    with col2:
        stamina = st.number_input('Stamina', min_value=1, max_value=99, value=75)
        strength = st.number_input('Strength', min_value=1, max_value=99, value=70)
        reputation = st.selectbox('International reputation', options=[1, 2, 3, 4, 5], index=0)
        is_on_loan = st.selectbox('Is on loan', options=[0, 1], index=0)
        is_free_agent = st.selectbox('Is free agent', options=[0, 1], index=0)
        contract_years = st.number_input('Contract years left', min_value=0, max_value=10, value=2)
        team_encoded = st.number_input('Team encoded value', min_value=0.0, max_value=30.0, value=15.0, step=0.1)
        fw_score = st.number_input('Forward Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        mid_score = st.number_input('Midfielder Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        def_score = st.number_input('Defender Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        gk_score = st.number_input('Goalkeeper Score', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    position = st.selectbox('Position category', options=list(POSITION_TO_ONE_HOT.keys()))
    one_hot = POSITION_TO_ONE_HOT[position]

    data = {
        'Age': age,
        'Potential': potential,
        'foot': foot,
        'Wage': wage,
        'Height_cm': height,
        'Weight_kg': weight,
        'Acceleration': acceleration,
        'Sprint speed': sprint_speed,
        'Agility': agility,
        'Balance': balance,
        'Stamina': stamina,
        'Strength': strength,
        'International reputation': reputation,
        'Is_On_Loan': is_on_loan,
        'Is_Free_Agent': is_free_agent,
        'Contract_Years_Left': contract_years,
        'Team_encoded': team_encoded,
        'Forward Score': fw_score,
        'Midfielder Score': mid_score,
        'Defender Score': def_score,
        'Goalkeeper Score': gk_score,
        'Position Category_Defender': one_hot[0],
        'Position Category_Forward': one_hot[1],
        'Position Category_Goalkeeper': one_hot[2],
        'Position Category_Midfielder': one_hot[3],
    }

    return pd.DataFrame([data], columns=FEATURES)


def build_raw_input(team_mapping: dict) -> pd.DataFrame:
    st.header('Raw feature input')
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=15, max_value=45, value=22)
        potential = st.number_input('Potential', min_value=40, max_value=99, value=80)
        foot = st.selectbox('Preferred foot', options=['Right', 'Left'])
        wage = st.number_input('Wage (€ raw)', min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
        height = st.number_input('Height (cm)', min_value=150, max_value=210, value=180)
        weight = st.number_input('Weight (kg)', min_value=55, max_value=110, value=75)
        acceleration = st.number_input('Acceleration', min_value=1, max_value=99, value=75)
        sprint_speed = st.number_input('Sprint speed', min_value=1, max_value=99, value=75)

    with col2:
        agility = st.number_input('Agility', min_value=1, max_value=99, value=75)
        balance = st.number_input('Balance', min_value=1, max_value=99, value=70)
        stamina = st.number_input('Stamina', min_value=1, max_value=99, value=75)
        strength = st.number_input('Strength', min_value=1, max_value=99, value=70)
        reputation = st.selectbox('International reputation', options=[1, 2, 3, 4, 5], index=0)
        is_on_loan = st.selectbox('Is on loan', options=[0, 1], index=0)
        is_free_agent = st.selectbox('Is free agent', options=[0, 1], index=0)
        contract_years = st.number_input('Contract years left', min_value=0, max_value=10, value=2)

    team = st.selectbox('Team', sorted(team_mapping.keys()), index=0)

    st.subheader('Position & Positional Performance Scores')
    col3, col4 = st.columns(2)
    with col3:
        fw_score = st.number_input('Forward Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        mid_score = st.number_input('Midfielder Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    with col4:
        def_score = st.number_input('Defender Score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        gk_score = st.number_input('Goalkeeper Score', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    position = st.selectbox('Primary position category', options=list(POSITION_TO_ONE_HOT.keys()))
    one_hot = POSITION_TO_ONE_HOT[position]

    raw = {
        'Age': age,
        'Potential': potential,
        'foot': foot,
        'Wage': wage,
        'Height_cm': height,
        'Weight_kg': weight,
        'Acceleration': acceleration,
        'Sprint speed': sprint_speed,
        'Agility': agility,
        'Balance': balance,
        'Stamina': stamina,
        'Strength': strength,
        'International reputation': reputation,
        'Is_On_Loan': is_on_loan,
        'Is_Free_Agent': is_free_agent,
        'Contract_Years_Left': contract_years,
        'Team': team,
        'Forward Score': fw_score,
        'Midfielder Score': mid_score,
        'Defender Score': def_score,
        'Goalkeeper Score': gk_score,
        'Position Category_Defender': one_hot[0],
        'Position Category_Forward': one_hot[1],
        'Position Category_Goalkeeper': one_hot[2],
        'Position Category_Midfielder': one_hot[3],
    }

    return preprocess_raw_input(raw, team_mapping)


def predict_value(model, input_df: pd.DataFrame):
    prediction = model.predict(input_df)[0]
    natural_value = np.expm1(prediction)
    return prediction, natural_value


def show_player_lookup(df_raw: pd.DataFrame, df_processed: pd.DataFrame, model):
    st.header('Player lookup')
    search_term = st.text_input('Search by player name or team')

    if search_term:
        filtered = df_raw[
            df_raw['Name'].str.contains(search_term, case=False, na=False)
            | df_raw['Team'].str.contains(search_term, case=False, na=False)
        ]
    else:
        filtered = df_raw.copy()

    if filtered.empty:
        st.warning('No matches found. Try a different name or team.')
        return

    selected_name = st.selectbox('Choose a player', filtered['Name'].tolist())
    row = filtered[filtered['Name'] == selected_name].iloc[0]

    st.subheader('Player details')
    display_cols = [col for col in ['Name', 'Team', 'Best position', 'Age', 'Potential', 'Value', 'Wage'] if col in row]
    st.write(row[display_cols].to_frame().T)

    if model is not None and st.button('Predict selected player valuation'):
        idx = row.name
        input_df = df_processed.loc[[idx], FEATURES]
        pred_log, pred_natural = predict_value(model, input_df)

        st.success(f'**Predicted Market Value:** {format_currency(pred_natural)}')
        st.metric(label='Log-transformed Value', value=f'{pred_log:.4f}')

        with st.expander('View raw feature array fed to model'):
            st.dataframe(input_df.T)


def main():
    st.set_page_config(page_title='Football Player Valuation App', page_icon='⚽', layout='wide')

    st.title('⚽ Football Player Valuation App')
    st.markdown(
        'Predict market transfer values for professional players using Machine Learning regression models.'
    )

    try:
        df_raw, df_processed, team_mapping = load_data()
    except Exception as e:
        st.error(f'Error loading datasets: {e}')
        return

    model_name = st.sidebar.selectbox('Choose model', list(MODEL_FILES.keys()))
    model = load_model(model_name)

    st.sidebar.markdown('---')
    if model is not None:
        st.sidebar.write('**Model Info:**')
        st.sidebar.write(f'Class: `{type(model).__name__}`')
        st.sidebar.write(f'Features Expected: `{getattr(model, "n_features_in_", "N/A")}`')

    tabs = st.tabs(['🔍 Player Lookup', '⚙️ Custom Input', 'ℹ️ About'])

    with tabs[0]:
        show_player_lookup(df_raw, df_processed, model)

    with tabs[1]:
        st.header('Manual Prediction')
        input_mode = st.radio('Input Mode', ['Raw Features', 'Processed Features'], horizontal=True)

        if input_mode == 'Raw Features':
            input_df = build_raw_input(team_mapping)
        else:
            input_df = build_processed_input()

        st.subheader('Processed Model Feature Vector')
        st.dataframe(input_df)

        if model is not None and st.button('Predict Custom Value'):
            pred_log, pred_natural = predict_value(model, input_df)
            st.success(f'**Predicted Market Value:** {format_currency(pred_natural)}')
            st.metric(label='Log-transformed Value', value=f'{pred_log:.4f}')

            if input_mode == 'Raw Features':
                st.info('💡 Note: Raw wage was automatically log-transformed (`log1p`) before prediction.')

    with tabs[2]:
        st.header('About this application')
        st.markdown(
            """
            - **Data Pipelines:** Loads cleaned player records from `data/processed/data_after_feature_engineering.csv` and engineered features from `data/processed/final_feature_engineering.csv`.
            - **Model Serving:** Pickled models from `models/gb_model.pkl` and `models/stacked_model.pkl`.
            - **Target Variable:** Predicted values represent **log-transformed market value (`np.log1p(Value)`)**, which is transformed back to EUR via `np.expm1`.

            **Preprocessing Steps Applied:**
            1. `Wage` is log-transformed (`log1p`).
            2. `foot` is binary-encoded (`Right=1`, `Left=0`).
            3. `Team_encoded` is mapped using target-encoding values from `team_target_encoding.json`.
            4. `Position Category` is one-hot encoded across four major positions.
            """
        )
        st.code('streamlit run streamlit_app.py')


if __name__ == '__main__':
    main()
