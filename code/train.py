def train(model, epochs, train_all_losses, train_all_acc):
    model.train()
    # initial the running loss
    running_loss = 0.0
    # pick each data from trainloader i: batch index/ data: inputs and labels
    correct = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = torch.Tensor(labels)
        # print(type(labels))
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        # print statistics
        running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update parameters
        optimizer.step()
        
        result = outputs > 0.5
        correct += (result == labels).sum().item() 

        if i % 64 == 0: 
            print('Training set: [Epoch: %d, Data: %6d] Loss: %.3f' %
                  (epochs + 1, i * 64, loss.item()))
 
    acc = correct / (split_train * 40)
    running_loss /= len(trainloader)
    train_all_losses.append(running_loss)
    train_all_acc.append(acc)
    print('\nTraining set: Epoch: %d, Accuracy: %.2f %%' % (epochs + 1, 100. * acc))


def validation(model, val_all_losses, val_all_acc, best_acc):
    model.eval()
    validation_loss = 0.0
    correct = 0
    for data, target in validloader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = model(data)

        validation_loss += criterion(output, target).item()

        result = output > 0.5
        correct += (result == target).sum().item()


    validation_loss /= len(validloader)
    acc = correct / (len(validloader) * 40)

    val_all_losses.append(validation_loss)
    val_all_acc.append(acc)

    
    print('\nValidation set: Average loss: {:.3f}, Accuracy: {:.2f}%)\n'
          .format(validation_loss, 100. * acc))
    
    return acc


def test(model, attr_acc, attr_name=attributes):
    test_loss = 0
    correct = 0
    pred = []
    for data, target in testloader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = model(data)
        test_loss += criterion(output, target).item()

        result = output > 0.5
        correct += (result == target).sum().item()
        compare = (result == target)
        pred.append(compare[0])


    test_loss /= len(testloader)
    acc = correct / (len(testloader) * 40)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * acc))
    
    for m in range(len(attr_name)):
        num = 0
        for n in range(len(pred)):
            if pred[n][m]:
                num += 1
        accuracy = num / len(pred)
        attr_acc.append(accuracy)

    for i in range(len(attr_acc)):
        print('Attribute: %s, Accuracy: %.3f' % (attr_name[i], attr_acc[i]))


