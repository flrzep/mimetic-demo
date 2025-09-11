# API

## POST /predict (backend)
- multipart/form-data field `file`
- returns: { success, predictions: [{class_id, confidence, label?}], processing_time }

## POST /predict-base64 (backend)
- body: { image: base64 }

## GET /health (backend)
- returns: overall status and Modal connectivity

## WS /ws (backend)
- simple echo/progress channel (extend as needed)
