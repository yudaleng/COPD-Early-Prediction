import argparse
import os

import torch

from utils.predict_utils import (
    load_spiro_encoder,
    preprocess_data,
    run_spiro_encoder,
    run_spiro_explainer,
    load_cb_model,
    run_spiro_predictor,
)


def parse_arguments(params=None):
    parser = argparse.ArgumentParser(description='Predict COPD probability.')

    # Data arguments
    parser.add_argument(
        '-data',
        type=str,
        help='Input data string or path to a .xlsx file containing the data.',
        required=False,
        default=(params['data'] if params else './data/sample.xlsx')
    )
    parser.add_argument(
        '-age',
        type=int,
        help='Age of the patient.',
        required=False,
        default=(params['age'] if params else 53)
    )
    parser.add_argument(
        '-sex',
        type=int,
        help='Sex of the patient.',
        required=False,
        default=(params['sex'] if params else 0)
    )
    parser.add_argument(
        '-smoke',
        type=int,
        help='Smoking status of the patient.',
        required=False,
        default=(params['smoke'] if params else 1)
    )

    # Model paths
    parser.add_argument(
        '-spiro_encoder_path',
        type=str,
        help='Path to the trained SpiroEncoder model file.',
        required=False,
        default="./weights/SpiroEncoder.pth"
    )
    parser.add_argument(
        '-spiro_explainer_path',
        type=str,
        help='Path to the trained SpiroExplainer model file.',
        required=False,
        default="./weights/SpiroExplainer.cbm"
    )
    parser.add_argument(
        '-spiro_predictor_path',
        type=str,
        help='Path to the trained SpiroPredictor model file.',
        required=False,
        default="./weights/SpiroPredictor.cbm"
    )

    # Thresholds and other settings
    parser.add_argument(
        '-spiroexplainer_threshold',
        type=float,
        help='Threshold for SpiroExplainer model.',
        required=False,
        default=0.1
    )
    parser.add_argument(
        '-num_threads',
        type=int,
        help='Number of threads to use for prediction.',
        required=False,
        default=8
    )
    parser.add_argument(
        '-device_str',
        type=str,
        help='Device to use for prediction.',
        required=False,
        default='cpu'
    )

    return parser.parse_args()


def main(params=None):
    args = parse_arguments(params)

    folders = ['weights']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

    torch.set_num_threads(args.num_threads)

    processed_data = preprocess_data(
        input_path=args.data, age=args.age, sex=args.sex, smoke=args.smoke
    )

    # Run SpiroEncoder
    spiro_encoder_original_result, attention_weights, all_input_x = run_spiro_encoder(
        model=load_spiro_encoder(device_str=args.device_str, model_path=args.spiro_encoder_path),
        data=processed_data,
        device=torch.device(args.device_str if torch.cuda.is_available() else "cpu")
    )

    # Run SpiroExplainer
    spiro_explainer_result, image_base64 = run_spiro_explainer(
        model=load_cb_model(model_path=args.spiro_explainer_path),
        data=processed_data,
        threshold=args.spiroexplainer_threshold,
        spiro_encoder_original_result=spiro_encoder_original_result,
        attention_weights=attention_weights,
        all_input_x=all_input_x,
        is_show=True
    )

    # Run SpiroPredictor if necessary
    spiro_predictor_result = {}
    if not spiro_explainer_result:
        spiro_predictor = run_spiro_predictor(
            model=load_cb_model(model_path=args.spiro_predictor_path),
            data=processed_data
        )
        spiro_predictor_result = spiro_predictor[0][1:6]

    print(f"COPD Detection: {spiro_explainer_result}")
    if not spiro_explainer_result:
        print(f"Future COPD Prediction: {spiro_predictor_result}")

    return spiro_explainer_result, spiro_predictor_result, image_base64


if __name__ == "__main__":
    main()
