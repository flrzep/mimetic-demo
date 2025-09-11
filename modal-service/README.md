# Modal GPU Service (Mock)

This is a placeholder service providing the same interface you will expose via Modal. Replace with actual Modal setup and deployment code.

- GET /health
- POST /predict { image: <base64> }

To run locally (mock):

```bash
pip install -r requirements.txt
uvicorn modal_service:app --reload --port 9000
```
