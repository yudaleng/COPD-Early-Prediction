import base64
import io
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from model.DeepSpiro import DeepSpiro, MyDataset

config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False,
    "font.size": 18,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
}
rcParams.update(config)
SPIRO_RECORD_SERIES_KEY = 'flow'


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def smooth(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    return smoothed_data


def compute_flow_volume_by_num_points(series, max_num_points, volume_scale=0.001, time_scale=0.01,
                                      max_interp_volume=6.58):
    volume = (series * volume_scale).astype(np.float32)

    flow = np.concatenate(([0.0], np.diff(volume) / time_scale))

    def right_pad_array(arr, pad_value, max_num_points):
        if len(arr) > max_num_points:
            return arr[:max_num_points]
        else:
            return np.pad(arr, (0, max_num_points - len(arr)), 'constant', constant_values=(pad_value,))

    padded_volume = right_pad_array(volume, 0, max_num_points)
    padded_flow = right_pad_array(flow, 0, max_num_points)

    monotonic_volume = np.maximum.accumulate(padded_volume)
    volume_interp_intervals = np.linspace(start=0, stop=max_interp_volume, num=max_num_points)
    flow_volume = np.interp(volume_interp_intervals, xp=monotonic_volume, fp=padded_flow, left=0, right=0)

    return volume, flow, flow_volume


def compute_fef(flow: np.ndarray, volume: np.ndarray, volume_max: float):
    flow_size = len(flow)
    assert flow_size == len(volume), 'Flow and Volume lengths do not match.'
    assert flow_size > 1, 'Flow should have more than one values'
    volumes_over_25 = volume >= (0.25 * volume_max)
    volumes_over_50 = volume >= (0.50 * volume_max)
    volumes_over_75 = volume >= (0.75 * volume_max)
    if not any(volumes_over_75):
        raise ValueError(f'Cannot find FEF75 in volume curve: {volume}')

    idx_25 = np.argmax(volumes_over_25)
    idx_50 = np.argmax(volumes_over_50)
    idx_75 = np.argmax(volumes_over_75)
    assert 0 <= idx_25 <= idx_50 <= idx_75 < flow_size

    fef25, fef50, fef75 = flow[[idx_25, idx_50, idx_75]]
    fef25_75 = flow[idx_25: (idx_75 + 1)].mean()
    return fef25, fef50, fef75, fef25_75


def calculate_index(row):
    flow = row['flow_volume']
    last_index = np.argmin(flow[5:])
    PEF_index = len(flow) - 1 - np.argmax(flow[::-1])
    flow_index = len(flow[:last_index + 1])
    index_25 = int(0.25 * flow_index)
    index_50 = int(0.50 * flow_index)
    index_75 = int(0.75 * flow_index)

    return PEF_index, index_25, index_50, index_75, last_index


def calculate_acceleration(row):
    if 'flow_volume' in row and isinstance(row['flow_volume'], np.ndarray):
        flow_volume = row['flow_volume']
        try:
            index_pef = int(row['index_pef'])
            start_index_25 = int(row['index_fef25'])
            end_index_50 = int(row['index_fef50'])
            end_index_75 = int(row['index_fef75'])
            last_index = int(row['last_index'])
            if not (index_pef < start_index_25 < end_index_50 < end_index_75):
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        except ValueError:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        def calc_area_under_curve(flow_volume, start_index, end_index):
            slope = (flow_volume[end_index] - flow_volume[start_index]) / (end_index - start_index)
            intercept = flow_volume[start_index] - slope * start_index
            baseline = slope * np.arange(start_index, end_index + 1) + intercept
            differences = -(flow_volume[start_index:end_index + 1] - baseline)
            area_below = np.sum(differences[differences < 0]) * 0.01
            area_above = np.sum(differences[differences > 0]) * 0.01
            adjusted_area = area_below + area_above
            return adjusted_area

        area_pef_25 = calc_area_under_curve(flow_volume, index_pef, start_index_25)
        area_25_50 = calc_area_under_curve(flow_volume, start_index_25, end_index_50)
        area_50_75 = calc_area_under_curve(flow_volume, end_index_50, end_index_75)
        area_75 = calc_area_under_curve(flow_volume, end_index_75, last_index - 1)
        area_pef_75 = calc_area_under_curve(flow_volume, start_index_25, last_index - 1)
        area_p = area_pef_25 + area_25_50 - area_50_75 - area_75
        return area_pef_25, area_25_50, area_50_75, area_75, area_pef_75, area_p


def process_data(row):
    handle = row.copy()
    series = [int(v) for v in row[SPIRO_RECORD_SERIES_KEY].split(',')]
    series = np.array(series)
    series = smooth(series)
    volume, flow, flow_volume = compute_flow_volume_by_num_points(series, len(series))
    fef25, fef50, fef75, fef25_75 = compute_fef(flow, volume, volume.max())
    handle['blow_fef25'] = fef25
    handle['blow_fef50'] = fef50
    handle['blow_fef75'] = fef75
    handle['blow_fef25_75'] = fef25_75
    handle['flow_volume'] = flow_volume
    handle['volume'] = volume
    handle['flow'] = flow
    handle['series'] = series
    handle['PEF'] = row['pef'] if row['pef'] != '' else np.nan
    handle["FEV1"] = row["fev1"] if row["fev1"] != '' else np.nan
    handle["FVC"] = row["fvc"] if row["fvc"] != '' else np.nan
    return handle


def process_acceleration(row):
    row['index_pef'], row['index_fef25'], row['index_fef50'], row['index_fef75'], row['last_index'] = calculate_index(
        row)
    acceleration_pef_25, acceleration_25_50, acceleration_50_75, acceleration_75, acceleration_pef_75, acceleration_total = calculate_acceleration(
        row)
    acceleration_pef_25 = pd.Series(acceleration_pef_25)
    acceleration_25_50 = pd.Series(acceleration_25_50)
    acceleration_50_75 = pd.Series(acceleration_50_75)
    acceleration_75 = pd.Series(acceleration_75)
    acceleration_pef_75 = pd.Series(acceleration_pef_75)
    acceleration_total = pd.Series(acceleration_total)
    row['PEF_FEF25'] = acceleration_pef_25
    row['FEF25_FEF50'] = acceleration_25_50
    row['FEF50_FEF75'] = acceleration_50_75
    row['FEF75'] = acceleration_75
    row['PEF_FEF75'] = acceleration_pef_75
    row['TOTAL'] = acceleration_total
    return row


def preprocess_data(input_path, age, sex, smoke):
    if input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
        if len(df) == 1:
            row = df.iloc[0]
            row = process_data(row)
            row = process_acceleration(row)
            processed_data = pd.Series(dtype='float64')
            processed_data['flow_volume'] = row['flow_volume']
            processed_data['PEF_FEF25'] = row['PEF_FEF25'].values[0]
            processed_data['FEF25_FEF50'] = row['FEF25_FEF50'].values[0]
            processed_data['FEF50_FEF75'] = row['FEF50_FEF75'].values[0]
            processed_data['FEF75'] = row['FEF75'].values[0]
            processed_data['PEF_FEF75'] = row['PEF_FEF75'].values[0]
            processed_data['TOTAL'] = row['TOTAL'].values[0]
            processed_data['AGE'] = age
            processed_data['SEX'] = sex
            processed_data['smoke'] = smoke
            processed_data['blow_ratio'] = 1 - (row['FEV1'] / row['FVC'])
            processed_data['fef25'] = row['blow_fef25']
            processed_data['fef50'] = row['blow_fef50']
            processed_data['fef75'] = row['blow_fef75']
            processed_data['FEV1'] = row['FEV1']
            processed_data['FVC'] = row['FVC']
            return processed_data
        else:
            AssertionError("Error: Only one row of data is supported.")
    else:
        AssertionError("Error: Unsupported file format.")


def load_spiro_encoder(device_str, model_path):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = DeepSpiro(
        in_channels=1,
        out_channels=32,
        n_len_seg=30,
        n_classes=2,
        device=device,
        verbose=False
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


def load_cb_model(model_path):
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def run_spiro_encoder(model, data, device):
    dataset = MyDataset([data['flow_volume']], 30)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    predictions = []
    attention_weights = []
    all_input_x = []
    with torch.no_grad():
        for data, mask in data_loader:
            data = data.to(device)
            mask = mask.to(device)
            output = model(data, mask)
            probabilities = torch.softmax(output, dim=1)
            predictions.append(probabilities.cpu().numpy())

            temporal_attention_weights = model.temporal_attention.attention_weights.detach().cpu().numpy()
            temporal_attention_weights = np.squeeze(temporal_attention_weights, axis=-1)
            input_x1 = data.detach().cpu().numpy()
            input_x1 = input_x1.reshape((1, -1, 1))
            attention_weights_padded = np.zeros((1, data.shape[1], 1))
            attention_weights_padded[:, :temporal_attention_weights.shape[1], :] = temporal_attention_weights[:, :,
                                                                                   None]
            attention_weights_expanded = np.repeat(attention_weights_padded, 30, axis=1)
            attention_weights.append(attention_weights_expanded)
            all_input_x.append(input_x1)
    return predictions, attention_weights, all_input_x


def run_spiro_explainer(model, data, threshold, spiro_encoder_original_result, attention_weights, all_input_x,
                        is_show=True):
    spiro_encoder_result = spiro_encoder_original_result[0][0][1]
    buf = plt_attention(
        all_input_x, attention_weights, data['fef25'], data['fef50'],
        data['fef75'], data['FEV1'], data['FVC'], is_show=is_show
    )
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    data['copd_detection'] = spiro_encoder_result

    X_pred = [[data['AGE'], data['SEX'], data['smoke'], data['blow_ratio'], data['copd_detection']]]
    probabilities = model.predict_proba(X_pred)
    detection = (probabilities[0][1] >= threshold).astype(int)
    return detection, image_base64


def run_spiro_predictor(model, data):
    X_pred = [
        [data['PEF_FEF25'], data['FEF25_FEF50'], data['FEF50_FEF75'], data['FEF75'], data['PEF_FEF75'], data['TOTAL'],
         data['copd_detection']]]
    probabilities = model.predict_proba(X_pred)
    return probabilities


def find_closest_x(y_target, y_data):
    y_data = np.array(y_data)
    index = np.abs(y_data - y_target).argmin()
    return index


def plt_attention(input_x1, attention, fef25, fef50, fef75, fev1_value, fvc_value, is_show=True):
    input_x = input_x1[0][0, :, 0]
    attention_weights = attention[0][0, :, 0]
    length_actual = len(input_x)
    input_x = input_x[:int(length_actual)]
    attention_weights = attention_weights[:int(length_actual)]
    y_data = input_x
    pef_max = np.max(y_data)
    x_pef_max = np.argmax(y_data)
    x_pef25 = (x_pef_max + find_closest_x(fef25, y_data[x_pef_max:])) / 100.0
    x_pef50 = (x_pef_max + find_closest_x(fef50, y_data[x_pef_max:])) / 100.0
    x_pef75 = (x_pef_max + find_closest_x(fef75, y_data[x_pef_max:])) / 100.0
    x_pef_max = x_pef_max / 100.0

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.05, 12)
    ax.spines['bottom'].set_position(('data', -0.05))
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cmap = plt.get_cmap('Reds')
    attention_weights = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min())
    colors = cmap(attention_weights)
    for j in range(len(input_x) - 1):
        ax.plot([j / 100.0, (j + 1) / 100.0], [input_x[j], input_x[j + 1]], color=colors[j])
    max_y = ax.get_ylim()[1]
    for pef_value, pef_x, label, color in zip([fef25, fef50, fef75, pef_max],
                                              [x_pef25, x_pef50, x_pef75, x_pef_max],
                                              ['FEF25', 'FEF50', 'FEF75', 'PEF'],
                                              ['red', 'blue', 'green', 'purple']):
        ax.vlines(x=pef_x, ymin=0, ymax=pef_value, colors=color, linestyles='--', label=label)
        ax.text(pef_x + 0.1, pef_value, f'{label}', verticalalignment='bottom', horizontalalignment='left',
                color=color)
    ax.annotate(f'FEV1: {fev1_value:.2f}L', xy=(fev1_value + 0.15, 0.0),
                xytext=(fev1_value, -max_y * 0.15),
                textcoords='data',
                va='top',
                ha='center',
                fontsize=6,
                arrowprops=dict(facecolor='orange', shrink=0.05))
    ax.annotate(f'FVC: {fvc_value:.2f}L', xy=(fvc_value - 0.15, 0.0),
                xytext=(fvc_value, -max_y * 0.15),
                textcoords='data',
                va='top',
                ha='center',
                fontsize=6,
                arrowprops=dict(facecolor='grey', shrink=0.05))
    ax.legend()
    ax.set_xlabel('Volume(L)')
    ax.set_ylabel('Flow(L/s)')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attention_weights.min(), vmax=attention_weights.max()))
    sm.set_array([])

    fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    if is_show:
        plt.show()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
