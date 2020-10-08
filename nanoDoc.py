from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.model import build_five_mer_model
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results


if __name__ == "__main__":
    args = parse_args()
    x_train, y_train, x_test, y_test, x_val, y_val, standardize_scale = load_dna_data_vae(args.data_size, args.data_path, args.feature_scale)
    print(x_train.shape)
    # 5mer removal and add label
    five_mer_model = build_five_mer_model()
