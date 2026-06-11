import joblib
import numpy as np
from flask import Flask, request, jsonify

# charger le modèle au démarrage
bundle = joblib.load("modele.joblib")
modele = bundle["modele"]
scaler = bundle["scaler"]

NB_FEATURES = 30  # breast cancer : 30 features

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """Reçoit un JSON {"features": [...]}, renvoie {"prediction": ..., "proba": ..., "label": ...}.
    Valide que "features" existe et a la bonne longueur.
    """
    data = request.get_json()

    # cas limite : clé "features" manquante
    if not data or "features" not in data:
        return jsonify({"erreur": "Clé 'features' manquante"}), 400

    features = data["features"]

    # cas limite : mauvais nombre de valeurs
    if len(features) != NB_FEATURES:
        return jsonify({
            "erreur": f"Attendu {NB_FEATURES} features, reçu {len(features)}"
        }), 400

    # cas adversarial : valeurs non numériques
    try:
        features = [float(f) for f in features]
    except (ValueError, TypeError):
        return jsonify({"erreur": "Toutes les features doivent être numériques"}), 400

    # cas adversarial : tableau vide
    if len(features) == 0:
        return jsonify({"erreur": "Le tableau features est vide"}), 400

    # normaliser et prédire
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred  = int(modele.predict(X_scaled)[0])
    proba = modele.predict_proba(X_scaled)[0]
    label = "benigne" if pred == 1 else "maligne"

    return jsonify({
        "prediction": pred,
        "proba"     : round(float(proba[pred]), 3),
        "label"     : label
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modele": type(modele).__name__})

if __name__ == "__main__":
    app.run(debug=True, port=5000)