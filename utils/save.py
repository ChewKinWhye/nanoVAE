import os
import json


def create_directories(result_path):
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def save_results(output_filename, results, plt, encoder, predictor):
    result_path = os.path.join("results", output_filename)
    create_directories(result_path)

    encoder.save(os.path.join(result_path, "encoder.h5"))
    predictor.save(os.path.join(result_path, "predictor.h5"))
    
    plt.savefig(os.path.join(result_path, "Encoding_dimension.png"))
    
    result_json = {
        "Accuracy": round(results[0], 3),
        "Sensitivity": round(results[1], 3),
        "Specificity": round(results[2], 3),
        "Precision": round(results[3], 3),
        "AU-ROC": round(results[4], 3)}

    with open(f'{result_path}/result.json', 'w+') as outfile:
        json.dump(result_json, outfile, indent=4)
