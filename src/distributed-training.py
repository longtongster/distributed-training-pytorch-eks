import os

from io import BytesIO
from typing import Any, Callable

import boto3
import torch
import torch.distributed as dist

from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def setup_distributed():
    # if 'MASTER_ADDR' not in os.environ:
    #     # Running locally with torchrun
    #     os.environ['RANK'] = os.environ.get('LOCAL_RANK', '0')
    #     os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
    #     os.environ['MASTER_ADDR'] = '127.0.0.1'
    #     os.environ['MASTER_PORT'] = '29500'
    
    # Print debug info
    print(f"RANK: {os.environ.get('RANK')}", end="\n")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo'
    )

    
def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every batch from rank 0
        if batch % 10 == 0: #and dist.get_rank() == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"World size: {os.environ['WORLD_SIZE']} Global rank: {os.environ['RANK']}, local rank {os.environ['LOCAL_RANK']} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, loss_fn: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Gather results from all processes
    test_loss = torch.tensor(test_loss)#.cuda()
    correct = torch.tensor(correct)#.cuda()
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    test_loss /= num_batches
    correct /= size
    if dist.get_rank() == 0:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    setup_distributed()
    
    # Get local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Load datasets
    training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())

    # Create samplers for distributed training
    train_sampler = DistributedSampler(training_data, shuffle=True)
    test_sampler = DistributedSampler(test_data, shuffle=False)

    # Create dataloaders with distributed samplers
    loaded_train = DataLoader(
        training_data, 
        batch_size=64, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    loaded_test = DataLoader(
        test_data, 
        batch_size=64, 
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Create model and wrap it in DistributedDataParallel
    model = NeuralNetwork()
    model = DistributedDataParallel(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    epochs = 1
    for t in range(epochs):
        if dist.get_rank() == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        train_sampler.set_epoch(t)  # Important for proper shuffling
        train(loaded_train, model, loss_function, optimizer)
        test(loaded_test, model, loss_function)

    # Save model only on master process
    if dist.get_rank() == 0:
        # # save model to in-memory buffer
        # buffer = BytesIO()
        # torch.save(model.state_dict(), buffer)
        # buffer.seek(0)

        # # s3 details
        # bucket_name = "pytorch-svw"
        # s3_key = "trained-models/model.pth"

        # # Upload to s3
        # s3 = boto3.client("s3")
        # s3.upload_fileobj(buffer, bucket_name, s3_key)
        # print(f"Model directly uploaded to s3://{bucket_name}/{s3_key}")
        torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()