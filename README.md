# F1 Predictor

A small project that predicts Formula 1 race win probabilities using historical race results and live pre-race data.

The model is trained with XGBoost and primarily learns patterns related to grid position and recent driver performance.

The app automatically detects whether qualifying data is available. If not, the user can enter an expected starting grid before generating predictions.

---

## How to run

### Backend

pip install -r backend/requirements.txt
uvicorn backend.app:app --reload

### Frontend
cd f1-frontend
npm install
npm run dev

Backend runs at http://127.0.0.1:8000

Frontend runs at http://localhost:5173
