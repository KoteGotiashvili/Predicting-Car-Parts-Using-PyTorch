import torch.nn
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

# create dictionary of results
results = {"train_loss": [],
           "train_acc": [],
           "test_loss": [],
           "test_acc": []}
# Train step takes in a model and dataloader and trains the model on the dataloader
def step_train(model: torch.nn.Module,
          dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device=device
          ):
    # put the model in train mode
    model.train()

    # Setup train loss and train accuracy

    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X) # output model logits

        # Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero gradient
        optimizer.zero_grad()

        # Loss backward, backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrices to get average loss and accuracy per batch
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc


# Takes model and dataloader and evaluates the model on the dataloader
def step_test(model: torch.nn.Module,
          dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          device=device
          ):
    # Put model in evaluation mode
    model.eval()

    # Test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn inference mode(Remove unnecessary stuff)
    with torch.inference_mode():
        # Loop thorugh data batches
        for batch, (X,y) in enumerate(dataloader):
            # Sent data to the target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate loss
            loss=loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

        # Adjust metrics get average
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


def train(model: torch.nn.Module,
          test_dataloader: DataLoader,
          train_dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int=5 ,
          device=device):


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = step_train(model=model,
                                           dataloader = train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = step_test(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        # Print results
        print(f" Epoch {epoch} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} ")

        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the results
    return results




def get_results():
    return results

