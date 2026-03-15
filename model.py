import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from mol2vec.features import mol2alt_sentence, MolSentence
from gensim.models import word2vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

RANDOM_STATE = 41
TEST_SIZE = 0.2

# =============================================================================
# Build Mol2Vec Embeddings
# =============================================================================

def load_mol2vec_model(model_dir='models'):
    """Load pretrained Mol2Vec model. Downloads the model if it is not already not present."""
    model_path = Path(model_dir) / 'model_300dim.pkl'

    if not model_path.exists():
        print("  Downloading pretrained Mol2Vec model...")
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
        urllib.request.urlretrieve(url, model_path)
        print("  Done.")

    return word2vec.Word2Vec.load(str(model_path))

def mol_to_embedding(mol, mol2vec_model) -> np.ndarray | None:
    """Convert an RDKit mol object to a Mol2Vec embedding."""
    try:
        # use radius = 1 as a starting point
        sentence = mol2alt_sentence(mol, radius=1)
        embedding = np.zeros(300)
        count = 0
        for word in sentence:
            word_str = str(word)
            if word_str in mol2vec_model.wv:
                embedding += mol2vec_model.wv[word_str]
                count += 1
        if count > 0:
            #
            embedding /= count
            return embedding
        return None
    except:
        return None


def featurize(df: pd.DataFrame, mol2vec_model, smiles_col: str = 'SMILES'):
    """Convert a DataFrame of SMILES to Mol2Vec embeddings."""
    features, valid_indices, failed = [], [], []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])

        if mol is None:
            failed.append(idx)
            continue

        emb = mol_to_embedding(mol, mol2vec_model)
        if emb is not None:
            features.append(emb)
            valid_indices.append(idx)
        else:
            failed.append(idx)

    return np.array(features), valid_indices, failed

# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if label:
        print(f"\n{label}")
        print("-" * 40)
    print(f"  MAE:  {mae:.2f} °C")
    print(f"  RMSE: {rmse:.2f} °C")
    print(f"  R²:   {r2:.3f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_parity(y_true, y_pred, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)

    lims = [min(y_true.min(), y_pred.min()) - 10,
            max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, 'k--', alpha=0.75)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'MAE = {mae:.1f} °C\nR² = {r2:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =============================================================================
# Train
# =============================================================================

def train(data_path: str):
    """Train the model and save artifacts."""
    print("=" * 60)
    print("TRAINING — Mol2Vec + XGBoost")
    print("=" * 60)

    # Load Mol2Vec
    print("\n[1] Loading Mol2Vec model...")
    mol2vec_model = load_mol2vec_model()
    print(f"  Loaded (300-dim embeddings)")

    # Load data
    print("\n[2] Loading training data...")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    # Standardize column names
    smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
    tg_col = [c for c in df.columns if 'tg' in c.lower()][0]
    df = df.rename(columns={smiles_col: 'SMILES', tg_col: 'Tg'})
    # Clean Tg values
    df['Tg'] = pd.to_numeric(df['Tg'], errors='coerce')
    df = df.dropna(subset=['Tg'])
    print(f"  Loaded {len(df)} polymers")

    # Featurize
    print("\n[3] Featurizing (Mol2Vec embeddings)...")
    X, valid_indices, failed = featurize(df, mol2vec_model)
    y = df.loc[valid_indices, 'Tg'].values
    print(f"  Features: {X.shape[1]} dimensions")
    print(f"  Valid: {len(valid_indices)}, Failed: {len(failed)}")

    # Split
    print("\n[4] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale (don't think that this is necessary for XGBoost, but it's probably good practice and won't hurt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Early stopping validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.15, random_state=RANDOM_STATE
    )

    # Train (start with early stopping rounds of 30 to reduce risk of overfitting)
    print("\n[5] Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    # Evaluate
    print("\n[6] Evaluation...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    evaluate(y_train, y_pred_train, "Training Set")
    test_metrics = evaluate(y_test, y_pred_test, "Test Set")

    # Plot
    plot_parity(y_test, y_pred_test, 'Mol2Vec + XGBoost → Tg',
                save_path='parity_plot.png')

    # Save trained artifacts
    print("\n[7] Saving trained model...")
    artifacts = {
        'xgb_model': model,
        'scaler': scaler,
    }
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("  Saved to trained_model.pkl")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Embedding: Mol2Vec (300 dim)")
    print(f"  Model:     XGBoost")
    print(f"  Test R²:   {test_metrics['r2']:.3f}")
    print(f"  Test MAE:  {test_metrics['mae']:.2f} °C")
    print("=" * 60)


# =============================================================================
# Predict
# =============================================================================

def predict(data_path: str):
    """Load trained model and predict Tg for new SMILES."""
    print("=" * 60)
    print("PREDICTION — Mol2Vec + XGBoost")
    print("=" * 60)

    # Load artifacts
    print("\n[1] Loading trained model...")
    with open('trained_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts['xgb_model']
    scaler = artifacts['scaler']
    print("  Loaded trained_model.pkl")

    # Load Mol2Vec
    print("\n[2] Loading Mol2Vec model...")
    mol2vec_model = load_mol2vec_model()

    # Load new data
    print("\n[3] Loading data...")
    df = pd.read_csv(data_path)
    # Standardize column names if needed
    if 'Tg (C)' in df.columns:
        df = df.rename(columns={'Tg (C)': 'Tg'})
    print(f"  Loaded {len(df)} polymers")

    # Featurize
    print("\n[4] Featurizing...")
    X, valid_indices, failed = featurize(df, mol2vec_model)
    print(f"  Valid: {len(valid_indices)}, Failed: {len(failed)}")

    # Scale and predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Build output
    results = df.loc[valid_indices].copy()
    results['Tg_predicted'] = predictions

    # If Tg values exist, evaluate
    if 'Tg' in df.columns:
        y_true = df.loc[valid_indices, 'Tg'].values
        print("\n[5] Evaluation (true values found)...")
        evaluate(y_true, predictions, "Prediction Results")
        plot_parity(y_true, predictions, 'Predictions vs Actual Tg')

    # Save
    output_path = 'predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"\n  Predictions saved to {output_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    _CLI_MODE = True
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Polymer Tg Prediction')
    parser.add_argument('--train', type=str, help='Path to training data (CSV or Excel)')
    parser.add_argument('--predict', type=str, help='Path to CSV for prediction (must contain a SMILES column)')
    args = parser.parse_args()

    if args.train:
        train(args.train)
    elif args.predict:
        predict(args.predict)
    else:
        parser.print_help()
