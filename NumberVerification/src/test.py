from VerificationCode.data_analysis import load_data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable
from torch import load, from_numpy, max, argmax
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

BATCH_SIZE = 64
data_set = load_data()
test_dataset = TensorDataset(
    from_numpy(data_set[2]), from_numpy(data_set[3]))
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

cnn = load('F:/cnn.pkl')

# 定义损失函数
criterion = CrossEntropyLoss()

cnn.eval()
eval_loss = 0
eval_acc = 0
for images, labels in test_loader:
    images = images.unsqueeze(1)
    images = Variable(images)
    out = cnn(images)
    loss = criterion(out, labels)
    eval_loss += loss.data.item() * labels.size(0)
    _, pred = max(out, 1)
    num_correct = (pred == labels).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))

for images, labels in test_loader:
    images = images.unsqueeze(1)
    images = Variable(images)
    out = cnn(images)
    for image, label in zip(images, out):
        # print(image)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        print(argmax(label))
        plt.show()

