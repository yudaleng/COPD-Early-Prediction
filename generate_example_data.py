import os
import numpy as np
import pandas as pd
import requests


def fetch_example_volumes(url):
    response = requests.get(url)
    response.raise_for_status()
    example_volumes_data = response.text.strip()
    example_cleaned_content = ','.join(example_volumes_data.split(',')[2:])
    return np.array([int(v) for v in example_cleaned_content.split(',')])


def calculate_pulmonary_metrics(volumes):
    fev1 = volumes[int(1 / 0.01)] / 1000
    pef = (np.max(np.diff(volumes)) / 0.01) * 60 / 1000
    fvc = volumes[-1] / 1000
    return fev1, pef, fvc


def save_metrics_to_excel(fev1, pef, fvc, volumes, file_path="./data/sample.xlsx"):
    df = pd.DataFrame({
        'fvc': [fvc],
        'fev1': [fev1],
        'pef': [pef],
        'flow': [','.join(map(str, volumes.tolist()))]
    })
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_excel(file_path, index=False)


def main():
    example_dataset_url = "https://biobank.ndph.ox.ac.uk/showcase/ukb/examples/eg_spiro_3066.dat"
    example_volumes = fetch_example_volumes(example_dataset_url)
    fev1, pef, fvc = calculate_pulmonary_metrics(example_volumes)
    save_metrics_to_excel(fev1, pef, fvc, example_volumes)


if __name__ == "__main__":
    main()
