def init():
    global predictor
    from predictor import CPUPredictor
    predictor = CPUPredictor()
    predictor.load_models()

def run(raw_data):
    import json
    data = json.loads(raw_data)
    preds = predictor.predict_all(data)
    return preds
