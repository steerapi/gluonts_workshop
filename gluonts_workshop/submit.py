import matplotlib.pyplot as plt
import requests
import json

# define helper functions
def plot_single_timeseries(train_data, predictions, test_data, item_id):
    plt.figure(figsize=(15, 3))

    y_past = train_data.loc[item_id]["target"]
    y_pred = predictions.loc[item_id]
    y_test = test_data.loc[item_id]["target"][-24:]

    colors = ['#FF6347', '#00FF7F', '#4169E1', '#FF69B4']

    plt.plot(y_past[-100:], label="Past Time Series", color=colors[0], linestyle='-')
    plt.plot(y_pred["mean"], label="Forecast", color=colors[1], linestyle='-')
    plt.plot(y_test, label="Observed", color=colors[2], linestyle='--')

    plt.fill_between( y_pred.index, y_pred["0.1"], y_pred["0.9"], color=colors[3], alpha=0.2, label="10%-90% Confidence Interval")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Predictions for Item {item_id}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_multiple_timeseries(train_data, predictions, test_data, item_ids_to_plot):

    colors = ['#FF6347', '#00FF7F', '#4169E1', '#FF69B4']

    plt.figure(figsize=(13, 13))
    for i, item_id in enumerate(item_ids_to_plot):
        plt.subplot(5, 1, i + 1)

        y_past = train_data.loc[item_id]["target"]
        y_pred = predictions.loc[item_id]
        y_test = test_data.loc[item_id]["target"][-24:]

        plt.plot(y_past[-100:], label="Past Time Series", color=colors[0], linestyle='-')
        plt.plot(y_pred["mean"], label="Forecast", color=colors[1], linestyle='-')
        plt.plot(y_test, label="Observed", color=colors[2], linestyle='--')

        plt.fill_between(y_pred.index, y_pred["0.1"], y_pred["0.9"], color=colors[3], alpha=0.2, label="10%-90% Confidence Interval")

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Predictions for Item {item_id}")
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def get_single_timeseries_plot_base64(train_data, predictions, test_data, item_id):
    fig = plt.figure(figsize=(15, 3))

    y_past = train_data.loc[item_id]["target"]
    y_pred = predictions.loc[item_id]
    y_test = test_data.loc[item_id]["target"][-24:]

    colors = ['#FF6347', '#00FF7F', '#4169E1', '#FF69B4']

    plt.plot(y_past[-100:], label="Past Time Series",
             color=colors[0], linestyle='-')
    plt.plot(y_pred["mean"], label="Forecast", color=colors[1], linestyle='-')
    plt.plot(y_test, label="Observed", color=colors[2], linestyle='--')

    plt.fill_between(y_pred.index, y_pred["0.1"], y_pred["0.9"],
                     color=colors[3], alpha=0.2, label="10%-90% Confidence Interval")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Predictions for Item {item_id}")
    plt.grid(True, linestyle='--', alpha=0.6)

    # get base64 string of jpg image of the plot
    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf8')
    plt.close()
    return image_base64

def submit(name: str, predictor, train_data, test_data, known_covariates=None):
    print(f"Submitting {name} to the leaderboard...")
    model_info = predictor.info()['model_info'][predictor.model_best]

    # generate predictions
    predictions = predictor.predict(
        train_data,
        random_seed=16,
        known_covariates=known_covariates
    )
    # plot time series
    image_base64 = get_single_timeseries_plot_base64(
        train_data, predictions, test_data, 1)

    # submit data as post request
    url = 'https://example.com/'
    headers = {'Content-type': 'application/json'}
    data = {
        "name": name,
        "model": model_info,
        "plot": image_base64,
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print("Submitted successfully!")
