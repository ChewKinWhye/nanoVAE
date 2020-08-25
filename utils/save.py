import os
import json


def create_directories(result_path):
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def save_results(output_filename, results, encoder, predictor):
    result_path = f'./results/{output_filename}'
    create_directories(result_path)
    encoder.save(f'{result_path}/encoder.h5')
    predictor.save(f'{result_path}/predictor.h5')
    result_json = {
        "Best au_prc": round(results[0], 3),
        "Best au_roc": round(results[3], 3),
        "Best accuracy": round(results[6], 3),
        "Best F-Measure": round(results[7], 3)}

    with open(f'{result_path}/result.json', 'w+') as outfile:
        json.dump(result_json, outfile, indent=4)
