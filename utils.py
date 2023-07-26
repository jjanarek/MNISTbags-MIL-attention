import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from sklearn.metrics import matthews_corrcoef as mmc
from sklearn.metrics import roc_auc_score


def train(train_loader, model, optimizer, device='cpu'):
    """
    Basic training loop over the training set (loader) and performing optimizer step.

    Returns train_loss and train_error averaged over the train loader length.
    """
    model.train()
    train_loss, train_error = 0.0, 0.0

    for data, label in train_loader:
        optimizer.zero_grad()

        data, label = data.to(device), label.to(device)
        label = label.float()

        loss, _, _ = model.calculate_objective(data, label)  # returns also gamma and alpha (kernel)
        train_loss += loss.item()
        error, _, _ = model.calculate_classification_error(data, label)  # returns also gamma and alpha (kernel)
        train_error += error

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error


def evaluation(eval_loader, model, mode='validation', device='cpu'):
    """
    Basic evaluation loop over eval_dataset (loader). Returns evaluation loss and error averaged over eval_dataset.
    If mode == 'test' returns also probabilities, predictions of the model, and true labels.
    """
    model.eval()

    val_loss, val_error = 0., 0.

    y_pred_list = []
    y_hat_list = []
    true_labels = []

    for data, label in eval_loader:
        data, label = data.to(device), label.to(device)
        # label = label.float()

        loss, _, _ = model.calculate_objective(data, label)  # returns also gamma and alpha (kernel)
        val_loss += loss.item()
        error, _, _ = model.calculate_classification_error(data, label)  # returns also gamma and alpha (kernel)
        val_error += error

        if mode == "test":
            y_pred, y_hat = model.calculate_prediction(data, label)
            y_pred_list.append(y_pred.item())
            y_hat_list.append(y_hat.item())
            true_labels.append(label.item())

    val_error /= len(eval_loader)
    val_loss /= len(eval_loader)

    if mode == 'validation':
        return val_loss, val_error
    elif mode == 'test':
        return val_loss, val_error, y_pred_list, y_hat_list, true_labels


def experiment(args, model, optimizer, train_set, valid_set, test_set):
    """
    Basic function performing model training and testing for one fold. Saves the best model (early stopping based
    on error and loss on validation set) and saves the data (history of training and test results).
    """

    epochs = args.epochs
    device = args.device
    experiment_dir = args.fold_dir + "/" + str(args.model_name)
    min_valid_loss = np.inf
    min_valid_error = 1.0
    best_epoch_iter = 0

    train_error_history = []
    train_loss_history = []

    valid_error_history = []
    valid_loss_history = []

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        model, train_loss, train_error = train(train_loader, model, optimizer, device=device)
        valid_loss, valid_error = evaluation(valid_loader, model, mode='validation', device=device)

        train_error_history.append(train_error)
        train_loss_history.append(train_loss)
        valid_error_history.append(valid_error)
        valid_loss_history.append(valid_loss)

        if args.verbose:
            print(f"Epoch {epoch}/{epochs}:")
            print(f"\t\tTraining   error: {train_error:.4f}\t training loss:   {train_loss:.4f}")
            print(f"\t\tValidation error: {valid_error:.4f}\t validation loss: {valid_loss:.4f}")

        # Early stopping mechanism
        if valid_error < min_valid_error:
            best_epoch_iter = 0
            min_valid_error = valid_error
            min_valid_loss = valid_loss
            torch.save(model, str(experiment_dir) + ".model")

            if args.verbose:
                print("Model saved")

        elif valid_error == min_valid_error:
            if valid_loss < min_valid_loss:
                best_epoch_iter = 0
                min_valid_loss = valid_loss
                min_valid_error = valid_error
                torch.save(model, str(experiment_dir) + ".model")

                if args.verbose:
                    print("Model saved")
            else:
                best_epoch_iter += 1
        else:
            best_epoch_iter += 1

        if best_epoch_iter > args.patience:
            if args.verbose:
                print("Early stopping: no progress in validation error (and loss).")
            break

    best_model = torch.load(str(experiment_dir) + ".model")
    if args.verbose:
        print("Best model loaded")

    test_loss, test_error, y_pred_list, y_hat_list, true_labels = evaluation(test_loader,
                                                                             best_model,
                                                                             mode='test',
                                                                             device=device)
    output = dict()
    output["train_error_history"] = train_error_history
    output["train_loss_history"] = train_loss_history
    output["valid_error_history"] = valid_error_history
    output["valid_loss_history"] = valid_loss_history
    output["y_pred_list"] = y_pred_list
    output["y_hat_list"] = y_hat_list
    output["true_labels"] = true_labels

    fold_mmc = mmc(np.array(true_labels, dtype=float), np.array(y_hat_list, dtype=float))
    fold_auc = roc_auc_score(np.array(true_labels, dtype=float), np.array(y_pred_list, dtype=float))
    output["best_model_mmc"] = fold_mmc
    output["best_model_auc"] = fold_auc
    output["test_error"] = test_error
    output["test_loss"] = test_loss

    np.savez_compressed(str(experiment_dir)+"_results.npz", **output)

    if args.verbose:
        print("\nTest set results:")
        print(f"Test error: {test_error:.4f}, MMC: {fold_mmc:.4f}, AUC: {fold_auc:.4f}\n")

    return test_error, fold_mmc, fold_auc


def bag_plot(bag: tuple) -> None:
    """
    Bag plot helper
    """
    items, label = bag
    label = label.numpy() * 1
    items = items.numpy()
    n_items = len(items)

    cols = 5
    rows = n_items // cols + 1
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(4, 4))
    axs = iter(axs.flatten())

    for i in range(cols * rows):
        ax = next(axs)
        if i < n_items:
            item = np.squeeze(items[i])

            ax.imshow(item, cmap='gray')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        ax.set_axis_off()

    fig.suptitle(f"Label: {label}")
    fig.tight_layout()

    fig.show()


def bag_weights_plot(bag: tuple, weights) -> None:
    """
    Bag plot helper with weights
    """
    items, label = bag
    label = label.numpy() * 1
    items = items.numpy()
    n_items = len(items)

    cols = 5
    rows = n_items // cols + 1
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(4, 4))
    axs = iter(axs.flatten())

    for i in range(cols * rows):
        ax = next(axs)
        if i < n_items:
            item = np.squeeze(items[i])

            ax.imshow(item, cmap='gray')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.set_title("{:.3f}".format(weights[i]))
        ax.set_axis_off()

    fig.suptitle(f"Label: {label}")
    fig.tight_layout()

    fig.show()


