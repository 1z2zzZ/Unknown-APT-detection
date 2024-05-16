import torch

def test_model(model, test_data, threshold):
    model.eval()
    with torch.no_grad():
        z_test, reconstructions_test = model(test_data.x, test_data.edge_index)
        reconstruction_errors_test = torch.norm(test_data.x - reconstructions_test, dim=1)

    anomaly_threshold_test = reconstruction_errors_test.mean() + threshold * reconstruction_errors_test.std()
    predictions_test = (reconstruction_errors_test > anomaly_threshold_test).float()

    predictions_test = predictions_test.cpu()
    accuracy_test = torch.sum(predictions_test == test_data.y.cpu()).item() / len(test_data.y)

    return accuracy_test