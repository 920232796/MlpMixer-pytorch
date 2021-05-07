import torch
from MlpMixer.model import  MlpMixer
import torchvision
import torch
import torch.nn as nn
# !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# !tar -zxvf MNIST.tar.gz

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))


def compute_acc(model, test_dataloader):
    with torch.no_grad():
        right_num = 0
        total_num = 0
        for in_data, label in test_dataloader:
            in_data = in_data.to(device)
            label = label.to(device)
            total_num += len(in_data)
            out = model(in_data)
            pred = out.argmax(dim=-1)
            for i, each_pred in enumerate(pred):
                if int(each_pred) == int(label[i]):
                    right_num += 1

        return (right_num / total_num)


if __name__ == "__main__":

    model = MlpMixer(in_dim=1, hidden_dim=32,
                     mlp_token_dim=32, mlp_channel_dim=32,
                     patch_size=(7, 7), img_size=(28, 28),
                     num_block=2, num_class=10
                )
    print(model)
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=16,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=16,
                                                   shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    report_loss = 0
    step = 0
    best_acc = 0.0

    for in_data, label in data_loader_train:
        batch_size = len(in_data)
        in_data = in_data.to(device)
        label = label.to(device)
        # print(in_data.shape)
        # print(label.shape)
        optimizer.zero_grad()
        step += 1
        out = model(in_data)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()
        report_loss += loss.item()
        if step % 10 == 0:
            print("report_loss is : " + str(report_loss))
            report_loss = 0
            acc = compute_acc(model, data_loader_test)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "./mnist_model.pkl")

            print("acc is " + str(acc) + ", best acc is " + str(best_acc))
