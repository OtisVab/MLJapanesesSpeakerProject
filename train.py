import argparse
from tqdm import tqdm

import torch.optim as optim
import numpy as np

from dataset import SpeechData, SequenceDataLoader, SplitDataLoader

import models
from models import *

from utils import *

################################################################################

def train(
    model, dataset,
    learning_rate, batch_size, num_epochs,
    max_grad_norm,
    device,
    validate = None
):
    # Store all accuracies and losses per epoch
    model_loss, model_acc = [], []
    val_acc = []
    y_true, y_pred = [], []

    # Load data and create dataset object (to generate batches)
    data_loader = SequenceDataLoader(
        dataset,
        batch_size
    )

    # Setup the loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if len(list(model.parameters()))>0 else None

    # Run for all epochs
    epoch_bar = tqdm(range(num_epochs), desc = "Epoch loss: N/A accuracy: N/A", position=1)
    for epoch in epoch_bar:

        epoch_loss = 0
        epoch_acc = 0

        # Train for all batches
        batch_bar = tqdm(enumerate(data_loader), desc = "Batch loss: N/A accuracy: N/A", total=len(data_loader), position=0)
        for batch_nr, (inputs, targets) in batch_bar:

            inputs = tuple(x.to(device) for x in inputs)
            targets = targets.to(device)

            # Saving actual labels
            temp2 = (targets).tolist()
            y_true.extend(temp2)

            # Reset gradients
            if optimizer is not None:
                optimizer.zero_grad()

            # Forward step
            output = model(*inputs)

            # Saving predicted labels
            temp1 = torch.argmax(output, 1).tolist()
            y_pred.extend(temp1)

            # Clip gradients to limit overfitting
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)

            # Compute Loss
            loss = loss_fn(output, targets)

            if optimizer is not None:
                # Backpropagate gradient w.r.t. loss
                loss.backward()

                # Run optimizer step
                optimizer.step()

            # Compute and store current batch accuracy and loss
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_acc= (model.sample(output) == targets).float().sum()
            epoch_acc += batch_acc

            # Print loss and accuracy for current batch
            batch_bar.set_description(f"Batch loss: {batch_loss} accuracy: {batch_acc}")
            batch_bar.refresh()

        # Store mean loss and accuracy for the last epoch
        model_loss.append(epoch_loss/len(dataset))
        model_acc.append(epoch_acc/len(dataset))

        if validate is not None:
            val_acc.append(validate(model))

        # Print mean loss and accuracy for the last epoch
        epoch_bar.set_description(
            f"Epoch loss: {model_loss[-1]} accuracy: {model_acc[-1]}" + ("" if validate is None else f" val acc: {val_acc[-1]}")
        )
        epoch_bar.refresh()

    return model_loss, model_acc, val_acc, y_true, y_pred


def test(
    model, dataset,
    batch_size,
    device
):
    # Load data and create dataset object (to generate batches)
    data_loader = SequenceDataLoader(
        dataset,
        batch_size
    )

    tot_acc = 0

    # Train for all batches
    batch_bar = tqdm(enumerate(data_loader), desc = "Batch loss: N/A accuracy: N/A")
    for batch_nr, (inputs, targets) in batch_bar:

        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)

        # Forward step
        output = model(*inputs)

        batch_acc = (model.sample(output) == targets).float().sum()
        tot_acc += batch_acc

        # Print loss and accuracy for current batch
        batch_bar.set_description(f"Batch accuracy: {batch_acc}")
        batch_bar.refresh()

    return float(tot_acc/len(dataset))

################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model parameters
    # TODO

    # Dataset params
    parser.add_argument(
        '--basefile',
        type= lambda x : x if (not x.endswith('.train') and not x.endswith('.test')) else '.'.join(x.split('.')[:-1]),
        required = True,
        help = "Base name of datasets to be used for training and testing"
    )

    parser.add_argument(
        '--lpc_vals',
        type= int, default = 12,
        help = "Number of LPC components to be used"
    )

    parser.add_argument(
        '--block_size_train',
        type= int, nargs = "+", action = listToVal,
        default = 30,
        help = "Block size(s) for training set"
    )

    parser.add_argument(
        '--block_size_test',
        type= int, nargs = "+", action = listToVal,
        default = [31, 35, 88, 44, 29, 24, 40, 50, 29],
        help = "Block size(s) for training set"
    )

    # Model params
    parser.add_argument(
        '--model_config',
        required=True,
        type = lambda x : x if x.endswith('.json') else x + '.json',
        help = 'Name of config file for model'
    )

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=lambda x : None if x.lower()=="none" else float(x), default=5.0, help='Maximum norm to clip gradients')

    # Running params
    parser.add_argument(
        '--device',
        type = lambda x : torch.device(x) if torch.cuda.is_available() else torch.device("cpu"),
        default = "cpu",
        help = "Training device 'cpu' or 'cuda:0'. Set always to 'cpu' if cuda is not available"
    )

    parser.add_argument(
        '--k-folds',
        type = lambda x : None if x.lower()=="none" else int(x),
        default = None,
        help = "Number of folds to use in k-folds cross validation, set/leave to None to use simple hold out cross-validation"
    )

    parser.add_argument(
        '--plot',
        action = 'store_true',
        default = False,
        help = "Set this flag to show loss/acc plots after training"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Load datasets

    print('Loading training set...')
    training_set = SpeechData(
        filename=f"{args.basefile}.train",
        block_size=args.block_size_train, num_LPC_vals=args.lpc_vals,
    )
    print('done')

    if args.k_folds is None:

        print('Loading test set...')
        test_set = SpeechData(
            f"{args.basefile}.test",
            block_size=args.block_size_test, num_LPC_vals=args.lpc_vals,
        )
        print('done')

        splits = lambda : [(training_set, test_set)]

    else:
        folds_acc = []
        splits = lambda : SplitDataLoader(training_set, args.k_folds)

    for fold_idx, (training_set_k, test_set_k) in enumerate(splits()):

        # Create model

        print('Creating model...')
        model = file2obj(
            models, args.model_config,
            input_size=args.lpc_vals
        )
        print('done')

        # Train the model
        print('Training...')
        train_loss, train_acc, val_acc, y_true, y_pred = train(
            model=model, dataset=training_set_k,
            learning_rate=args.learning_rate, batch_size=args.batch_size, num_epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            validate = lambda model : test(
                model=model, dataset=test_set_k,
                batch_size=args.batch_size,
                device=args.device
            )
        )
        print('done')

        # Final testing

        fold_acc = test(
            model=model, dataset=test_set_k,
            batch_size=args.batch_size,
            device=args.device
        )

        if args.k_folds is not None:
            folds_acc.append(fold_acc)

        print("Final accuracy", "" if args.k_folds is None else f"fold {fold_idx}: ", fold_acc)

        if args.plot:
            from matplotlib import pyplot as plt
            import pandas as pd
            import seaborn as sn

            plt.figure()
            plt.plot(train_loss)
            plt.title('Training loss'+ (" " if args.k_folds is None else f" for fold {fold_idx}"))
            plt.xlabel('epochs')
            plt.ylabel('Cross Entropy loss')

            plt.figure()
            plt.plot(train_acc, label='training')
            plt.plot(val_acc, label='validation')
            plt.legend()
            plt.title('Accuracy during training' + (" " if args.k_folds is None else f" for fold {fold_idx}"))
            plt.ylabel('Accuracy (%)')
            plt.xlabel('epochs')
            plt.ylim([0, 1])

            # Confussion Matrix
            plt.figure()
            y_actu = pd.Series(y_true, name='Actual classes')
            y_pred = pd.Series(y_pred, name='Predicted classes')
            cm = pd.crosstab(y_actu, y_pred)
            cm_norm = cm / cm.sum(axis=1)
            cm_norm = cm_norm.round(decimals=3)
            labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            sn.heatmap(cm_norm, annot=True, vmin=0, vmax=1,
                       xticklabels=labels, yticklabels=labels, cmap="RdYlGn")
            plt.title('Confussion Matrix')
            plt.show()

            plt.show(block=False)

    if args.k_folds is not None:
        folds_acc = np.array(folds_acc)

        print(
            f"Accuracies : ",
            *map(
                lambda x : f"Fold: {x[0]} acc: {x[1]}",
                enumerate(folds_acc)
            )
        )

        print(f"Mean: {folds_acc.mean()}\tSTD: {folds_acc.std()}\tSTE: {folds_acc.std()/np.sqrt(folds_acc.size)}")
