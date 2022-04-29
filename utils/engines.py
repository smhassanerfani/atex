import torch
import time
import copy
import csv


def csv_writer(model_name, log_list, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["epoch", "train_loss",
                      "train_acc", "val_loss", "val_acc"]

    with open(f"./outputs/{model_name}/loss_acc_results.csv", 'w', newline='') as filehandler:
        fh_writer = csv.DictWriter(filehandler, fieldnames=fieldnames)

        fh_writer.writeheader()
        for item in log_list:
            fh_writer.writerow(item)


class AdjustLearningRate:
    num_of_iterations = 0

    def __init__(self, optimizer, base_lr, max_iter, power):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power

    def __call__(self, current_iter):
        lr = self.base_lr * ((1 - float(current_iter) / self.max_iter) ** self.power)
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = lr * 10

        return lr

def train_model(
        model,
        model_name,
        dataloader,
        criterion,
        optimizer,
        base_lr,
        pdlr=True,
        scheduler=None,
        num_epochs=30,
        lr_pp=0.9):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_list = list()
    lr = base_lr

    max_iter = num_epochs * len(dataloader["train"].dataset)
    lr_poly = AdjustLearningRate(optimizer, base_lr, max_iter, lr_pp)

    for epoch in range(num_epochs):

        log_dic = {}
        log_dic["epoch"] = epoch

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if ("drn-101" in model_name):
                        aux, outputs = model(inputs)
                        loss = criterion(outputs, labels) + 0.4 * \
                            criterion(aux, labels)
                    if ("googlenet" in model_name) and (phase == "train"):
                        outputs, aux2, aux1 = model(inputs)
                        loss = criterion(outputs, labels) + 0.2 * \
                            criterion(aux2, labels) + 0.2 * \
                            criterion(aux1, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':

                        # pdlr adjusts the lr based on iterations
                        if pdlr == True:
                            lr_poly.num_of_iterations += len(inputs)
                            lr = lr_poly(lr_poly.num_of_iterations)

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # scheduler adjusts the lr based on epochs
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

            print(f'{phase} loss:\t {epoch_loss:.4f}\t acc:\t {epoch_acc:.4f}\t lr:\t {lr:.6f}')

            if phase == 'train':
                log_dic["train_loss"] = epoch_loss
                log_dic["train_acc"] = epoch_acc.item()

            # deep copy the model
            if phase == 'val':
                log_dic["val_loss"] = epoch_loss
                log_dic["val_acc"] = epoch_acc.item()

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_acc": epoch_acc,
                    }
                    save_path = f"./outputs/{model_name}/model.pth"
                    torch.save(state, save_path)

        log_list.append(log_dic)
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.2%}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    csv_writer(model_name, log_list)

    return model
