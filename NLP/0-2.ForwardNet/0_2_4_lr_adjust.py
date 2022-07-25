import torch
import math
import torch.optim as optim
"""
参考文献
https://blog.zhujian.life/posts/78a36c78.html
https://zhuanlan.zhihu.com/p/50499794
https://zhuanlan.zhihu.com/p/104472245
"""


def mannual_lr():
    lr = 0.001
    gamma = 0.96
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    """
    调整学习率,一个是指数衰减, 还有一个是使用自定义规则的lambda衰减.
    衰减公式为 lr = lr * (gamma ** epoch)
    https://www.cnblogs.com/wanghui-garcia/p/10895397.html
    """
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    rule = lambda epoch: 0.96**epoch
    lrr = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)

    for epoch in range(40):
        optimizer.step()
        print('epoch:{}, {}'.format(epoch, lrr.get_last_lr()[0]))
        lrr.step()
        scheduler.step()


def find_lr(data_loader, model, criterion, optimizer, device, init_value=1e-8, final_value=10., beta=0.98):
    """

    :param data_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param device:
    :param init_value:
    :param final_value:
    :param beta:
    :return:
    """
    num = len(data_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in data_loader:
        batch_num += 1
        print('{}: {}'.format(batch_num, lr))

        # As before, get the loss for this mini-batch of inputs/outputs
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Do the SGD step
        loss.backward()
        optimizer.step()

        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


if __name__ == "__main__":
    mannual_lr()