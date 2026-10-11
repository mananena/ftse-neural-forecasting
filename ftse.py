"""Сравнение AR, RNN и LSTM: прогноз на одно следующее наблюдение."""
from pathlib import Path
import argparse
import json
import hashlib
import platform
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.ar_model import AutoReg

WINDOW = 30
MAX_EPOCHS = 150
PATIENCE = 20
SEEDS = (42, 7, 123)
DROP_START = pd.Timestamp('2025-04-03', tz='UTC')
DATASETS = {'daily': 'ftse_daily.csv', '4h': 'ftse_4h.csv'}


def load_data(path):
    data = pd.read_csv(path, thousands=',')
    data['Date'] = pd.to_datetime(data['Date'], utc=True, errors='raise')
    data['Close'] = pd.to_numeric(data['Close'], errors='raise')
    data = data.sort_values('Date').set_index('Date')
    if data.index.has_duplicates or data.index.isna().any():
        raise ValueError('Даты должны быть уникальными и заполненными')
    raw_rows = len(data)
    missing_dates = [str(date) for date in data.index[data.Close.isna()]]
    # Нет наблюдаемой цели: удаляем строку, не заполняя её будущими ценами.
    data = data.dropna(subset=['Close'])
    data.attrs.update(raw_rows=raw_rows, missing_close_dates=missing_dates)
    if not np.isfinite(data['Close']).all() or (data['Close'] <= 0).any():
        raise ValueError('Нужны конечные положительные цены')
    if len(data) < 120:
        raise ValueError('Недостаточно наблюдений для выбранного окна и разбиения')
    return data


def make_sequences(values, start, end, window=WINDOW):
    """Цель с индексом t использует только values[t-window:t]."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not window <= start < end <= len(values):
        raise ValueError('Некорректные границы последовательностей')
    x = np.stack([values[t-window:t] for t in range(start, end)])[:, :, None]
    y = values[start:end, None]
    return torch.from_numpy(x), torch.from_numpy(y)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError('Метрики требуют непустых массивов одной формы')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('В метриках обнаружены NaN или inf')
    error = actual - predicted
    return {'MSE': float(np.mean(error**2)),
            'MAE': float(np.mean(np.abs(error))),
            'RMSE': float(np.sqrt(np.mean(error**2)))}


class RecurrentModel(nn.Module):
    def __init__(self, kind, differenced=False):
        super().__init__()
        # Для разностей уменьшаем число параметров: выборка небольшая.
        hidden = 16 if differenced else 50
        layers = 1 if differenced or kind == 'RNN' else 2
        recurrent = nn.RNN if kind == 'RNN' else nn.LSTM
        self.recurrent = recurrent(1, hidden, num_layers=layers, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        # PyTorch по умолчанию создаёт нулевые h0 и c0 на нужном устройстве.
        output, _ = self.recurrent(x)
        return self.linear(output[:, -1, :])


def fit_network(kind, differenced, x, y, seed, epochs=MAX_EPOCHS, validation=None):
    torch.manual_seed(seed)
    model = RecurrentModel(kind, differenced)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    criterion = nn.MSELoss()
    history, best_epoch, best_loss, stale = [], 1, np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_x)
        record = {'epoch': epoch, 'train_MSE_scaled': total / len(x)}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(validation[0]), validation[1]).item()
            record['validation_MSE_scaled'] = val_loss
            if val_loss < best_loss - 1e-7:
                best_epoch, best_loss, stale = epoch, val_loss, 0
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= PATIENCE:
            break
    # При подборе возвращаем номер лучшей эпохи; финальная модель учится заново.
    return model, best_epoch, history


def neural_forecast(prices, fit_end, test_start, kind, differenced, seed):
    prices = np.asarray(prices, dtype=float)
    series = np.diff(prices) if differenced else prices
    offset = int(differenced)
    fit_boundary, test_boundary = fit_end-offset, test_start-offset
    scaler = StandardScaler() if differenced else MinMaxScaler()
    scaler.fit(series[:fit_boundary, None])
    scaled = scaler.transform(series[:, None]).ravel()
    x, y = make_sequences(scaled, WINDOW, fit_boundary)
    xv, yv = make_sequences(scaled, fit_boundary, test_boundary)
    _, best_epoch, history = fit_network(kind, differenced, x, y, seed,
                                        validation=(xv, yv))
    # После выбора эпох используем все первые 80% данных. Test остаётся закрытым.
    scaler.fit(series[:test_boundary, None])
    scaled = scaler.transform(series[:, None]).ravel()
    x, y = make_sequences(scaled, WINDOW, test_boundary)
    model, _, _ = fit_network(kind, differenced, x, y, seed, epochs=best_epoch)
    xt, _ = make_sequences(scaled, test_boundary, len(series))
    model.eval()
    with torch.no_grad():
        prediction = scaler.inverse_transform(model(xt).numpy()).ravel()
    if differenced:
        prediction += prices[test_start-1:-1]
    return prediction, best_epoch, history


def ar_forecast(prices, start, lags, differenced=False, expanding=False):
    """Сначала прогнозируем, затем добавляем фактическую следующую цену."""
    prices = np.asarray(prices, dtype=float)
    series = np.diff(prices) if differenced else prices
    boundary = start-int(differenced)
    history = list(series[:boundary])
    fitted = AutoReg(np.asarray(history), lags=lags, trend='c', old_names=False).fit()
    predictions = []
    for t in range(boundary, len(series)):
        if expanding and t > boundary:
            fitted = AutoReg(np.asarray(history), lags=lags, trend='c', old_names=False).fit()
        parameters = np.asarray(fitted.params)
        next_value = parameters[0] + parameters[1:] @ np.asarray(history[-lags:][::-1])
        if differenced:
            next_value += prices[t]
        predictions.append(next_value)
        history.append(series[t])
    return np.asarray(predictions)


def select_lags(prices, fit_end, test_start, differenced):
    # Каждый кандидат оценивается только на отрезке validation.
    scores = {}
    for lag in (1, 5, 10, 30):
        predicted = ar_forecast(prices[:test_start], fit_end, lag, differenced)
        scores[lag] = metrics(prices[fit_end:test_start], predicted)['RMSE']
    return min(scores, key=scores.get), scores


def plot_results(data, prediction, losses, dataset, out):
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data.index, data.Close, lw=1.5)
    ax.axvline(prediction.index[0], color='gray', ls='--', label='Начало теста')
    ax.axvspan(DROP_START, data.index[-1], color='red', alpha=.12, label='3–8 апреля')
    ax.set(title=f'FTSE: {dataset}, исходные данные', ylabel='Пункты индекса', xlabel='Дата (UTC)')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / f'{dataset}_series.png'); plt.close(fig)

    groups = [('AR', 'AR_diff', 'AR_diff_expanding'), ('RNN', 'LSTM'), ('RNN_diff', 'LSTM_diff')]
    for period, frame in [('test', prediction), ('drop', prediction.loc[DROP_START:])]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, names in zip(axes, groups):
            ax.plot(frame.index, frame.actual, color='black', lw=2, label='Факт')
            ax.plot(frame.index, frame.Naive, color='gray', ls=':', label='Последняя цена')
            for name in names:
                ax.plot(frame.index, frame[name], label=name, lw=1.3)
            ax.set_ylabel('Пункты индекса'); ax.legend(ncol=3); ax.grid(alpha=.25)
        axes[0].set_title(f'FTSE {dataset}: {period}, прогноз на один шаг; нейросети seed=42')
        axes[-1].set_xlabel('Дата (UTC)')
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(out / f'{dataset}_{period}.png'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, history) in zip(axes.flat, losses.items()):
        frame = pd.DataFrame(history)
        ax.plot(frame.epoch, frame.train_MSE_scaled, label='Train')
        ax.plot(frame.epoch, frame.validation_MSE_scaled, label='Validation')
        ax.set(title=name, xlabel='Эпоха', ylabel='MSE в масштабе модели')
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f'Подбор числа эпох: {dataset}, seed=42')
    fig.tight_layout(); fig.savefig(out / f'{dataset}_loss.png'); plt.close(fig)


def run_experiment(data_dir=Path('data'), out=Path('results')):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_metrics, audits, settings = [], {}, {}
    for dataset, filename in DATASETS.items():
        path = data_dir / filename
        data = load_data(path)
        prices = data.Close.to_numpy()
        fit_end, test_start = int(.6*len(data)), int(.8*len(data))
        dates = data.index[test_start:]
        prediction = pd.DataFrame({'actual': prices[test_start:], 'Naive': prices[test_start-1:-1]}, index=dates)
        audits[dataset] = {'raw_rows': data.attrs['raw_rows'], 'missing_close_dates': data.attrs['missing_close_dates'],
                           'rows': len(data), 'first_utc': str(data.index[0]), 'last_utc': str(data.index[-1]),
                           'fit_rows': fit_end, 'validation_rows': test_start-fit_end,
                           'test_rows': len(data)-test_start, 'test_start_utc': str(dates[0]),
                           'drop_test_rows': int((dates >= DROP_START).sum()),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        settings[dataset] = {}
        for differenced in (False, True):
            name = 'AR_diff' if differenced else 'AR'
            lag, scores = select_lags(prices, fit_end, test_start, differenced)
            settings[dataset][name] = {'lags': lag, 'validation_RMSE': scores}
            prediction[name] = ar_forecast(prices, test_start, lag, differenced)
            if differenced:
                prediction['AR_diff_expanding'] = ar_forecast(prices, test_start, lag, True, True)
        def save_metrics(name, values, seed):
            for period, mask in [('test', np.ones(len(dates), dtype=bool)), ('drop', dates >= DROP_START)]:
                if mask.any():
                    all_metrics.append({'dataset': dataset, 'model': name, 'seed': seed,
                                        'period': period, 'n': int(mask.sum()),
                                        **metrics(prices[test_start:][mask], np.asarray(values)[mask])})
        for name in prediction.columns.drop('actual'):
            save_metrics(name, prediction[name], -1)
        losses = {}
        neural_predictions = []
        for differenced in (False, True):
            for kind in ('RNN', 'LSTM'):
                name = kind + ('_diff' if differenced else '')
                settings[dataset][name] = {}
                for seed in SEEDS:
                    values, epoch, history = neural_forecast(prices, fit_end, test_start, kind, differenced, seed)
                    save_metrics(name, values, seed)
                    settings[dataset][name][str(seed)] = {'best_epoch': epoch}
                    neural_predictions.extend({'Date': str(date), 'model': name, 'seed': seed, 'prediction': float(value)}
                                              for date, value in zip(dates, values))
                    if seed == 42:
                        prediction[name], losses[name] = values, history
                    print(f'{dataset}: {name}, seed={seed}, epochs={epoch}', flush=True)
        prediction.to_csv(out / f'{dataset}_predictions.csv', index_label='Date')
        pd.DataFrame(neural_predictions).to_csv(out / f'{dataset}_all_seeds.csv', index=False)
        (out / f'{dataset}_loss.json').write_text(json.dumps(losses, indent=2), encoding='utf-8')
        plot_results(data, prediction, losses, dataset, out)
    table = pd.DataFrame(all_metrics)
    table.to_csv(out / 'metrics.csv', index=False)
    summary = table.groupby(['dataset', 'period', 'model']).agg(
        n=('n', 'first'), MSE_mean=('MSE', 'mean'), MAE_mean=('MAE', 'mean'),
        RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', lambda x: x.std(ddof=0)), runs=('seed', 'size')).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    metadata = {'protocol': 'chronological 60/20/20; validation selection; refit first 80%; one-step test',
                'window': WINDOW, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                'seeds': SEEDS, 'drop_start_utc': str(DROP_START), 'data': audits, 'settings': settings,
                'python': platform.python_version(), 'platform': platform.platform(),
                'versions': {p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','matplotlib','scikit-learn','statsmodels']}}
    (out / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return table, summary, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    _, summary, _ = run_experiment(args.data_dir, args.out)
    print(summary.to_string(index=False))
