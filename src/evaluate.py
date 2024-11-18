import torch
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, criterion, label_encoder, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        test_loss = sum([criterion(outputs[:, i, :], y_test[:, i]) for i in range(outputs.size(1))])

        test_predicted = torch.argmax(outputs, dim=2)
        test_correct = (test_predicted == y_test).sum().item()
        test_total = y_test.size(0) * outputs.size(1)
        test_accuracy = 100 * test_correct / test_total

        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%')

        y_test_flat = y_test.flatten().cpu().numpy()
        test_predicted_flat = test_predicted.flatten().cpu().numpy()

        print(classification_report(y_test_flat, test_predicted_flat, target_names=label_encoder))
