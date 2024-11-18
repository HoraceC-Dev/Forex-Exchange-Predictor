import torch

def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs, device):
    best_model_state = None
    best_val_accuracy = 0
    val_loss_data = []
    train_loss_data = []
    learning_rates = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = sum([criterion(outputs[:, i, :], y_train[:, i]) for i in range(outputs.size(1))])
        train_loss_data.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #learning_rates.append(optimizer.param_groups[0]['lr'])
        #scheduler.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = sum([criterion(val_outputs[:, i, :], y_val[:, i]) for i in range(val_outputs.size(1))])
            val_loss_data.append(val_loss.item())
            val_predicted = torch.argmax(val_outputs, dim=2)
            val_correct = (val_predicted == y_val).sum().item()
            val_total = y_val.size(0) * val_outputs.size(1)
            val_accuracy = 100 * val_correct / val_total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return best_model_state,val_loss_data,train_loss_data,learning_rates
