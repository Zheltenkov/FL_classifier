import sys
import torch
from sklearn.metrics import f1_score


def train_loop(epoch_n, model, train_loader, criterion, optimizer, device, train_writer, num_classes):
    """
    :param epoch_n:
    :param model:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param device:
    :param train_writer:
    :param num_classes:
    :return:
    """
    print('-------------MODELS IN TRAIN STATUS-------------')
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size

    train_loss, train_accuracy, train_f1_score = 0, 0, 0

    model.train()

    for i, data in enumerate(iter(train_loader)):
        image, label = data['image'].to(device), data['label'].to(device)
        prediction_class = model(image)

        loss = criterion(prediction_class, label)

        # ZEROING GRADIENTS
        optimizer.zero_grad()
        loss.backward()

        # OPTIMIZE STEP
        optimizer.step()

        prediction = torch.argmax(prediction_class, dim=-1).cpu()
        print(prediction)
        sys.exit()


        labels = label.cpu()

        f1_scr = f1_score(prediction, labels, average='weighted')
        correct = (prediction_class.argmax(1).cpu() == label.cpu()).sum().item()
        accuracy = correct / batch_size

        # ADD LOSS/ACCURACY TO VALUES FOR THE EPOCH
        train_loss += loss.item()
        train_accuracy += accuracy
        train_f1_score += f1_scr

        if i != 0 and i % 2 == 0:
            print('----------LOGGING TRAIN BATCH-----------')
            print(f'Epoch - {epoch_n}, Batch - {i}, Total batch - {num_batches}')
            print('losses/total_loss', train_loss / i)
            print('accuracy/total_accuracy', train_accuracy / i)
            print('train_f1_score/total_train_f1_score', train_f1_score / i)
            print(40 * '--')


    # GET AVERAGE
    train_loss /= num_batches
    train_accuracy /= num_batches
    train_f1_score /= num_batches

    train_writer.add_scalar('losses/total_loss', train_loss, epoch_n)
    train_writer.add_scalar('accuracy/total_accuracy', train_accuracy, epoch_n)
    train_writer.add_scalar('train_f1_score/total_train_f1_score', train_f1_score, epoch_n)
    print(40 * '++')
    print('Train Metrics  on epoch - ' + str(epoch_n))
    print('Accuracy Metrics  on epoch - ' + str(train_accuracy))
    print('Avg loss  on epoch - ' + str(train_loss))
    print('F1 - score - ' + str(train_f1_score))
    print(40*'++')
    print(f'Train Metrics  on epoch {epoch_n}: \n Accuracy: {(train_accuracy):>0.3f}, Avg loss: {train_loss:>8f} \n'
          f'F1 - score: {(train_f1_score):>0.3f}')

    return train_loss, train_accuracy, train_f1_score


def valid_loop(epoch_n, model, valid_loader, criterion, device, valid_writer, num_classes):
    """
    :param epoch_n:
    :param model:
    :param valid_loader:
    :param criterion:
    :param device:
    :param valid_writer:
    :param num_classes:
    :return:
    """
    print("------------------MODELS IN VALIDATION STATUS-----------------")
    num_batches = len(valid_loader)
    batch_size = valid_loader.batch_size

    model.eval()

    valid_loss, valid_accuracy, valid_f1_score = 0, 0, 0

    with torch.no_grad():
        for i, data in enumerate(iter(valid_loader)):
            image, label = data['image'].to(device), data['label'].to(device)
            prediction_class = model(image)

            loss = criterion(prediction_class, label)

            prediction = torch.argmax(prediction_class, dim=-1).cpu()
            labels = label.cpu()

            f1_scr = f1_score(prediction, labels, average='weighted')
            correct = (prediction_class.argmax(1).cpu() == label.cpu()).sum().item()
            accuracy = correct / batch_size


            # ADD LOSS/ACCURACY TO VALUES FOR THE EPOCH
            valid_loss += loss.item()
            valid_accuracy += accuracy
            valid_f1_score += f1_scr

            if i != 0 and i % 2 == 0:
                print('----------LOGGING VALID BATCH-----------')
                print(f'Epoch - {epoch_n}, Batch - {i}, Total batch - {num_batches}')
                print('losses/total_loss', valid_loss/i)
                print('accuracy/total_accuracy', valid_accuracy/i)
                print('valid_f1_score/total_valid_f1_score', valid_f1_score/i)
                print(40 * '--')


        # GET AVERAGE
        valid_loss /= num_batches
        valid_accuracy /= num_batches
        valid_f1_score /= num_batches

        torch.cuda.empty_cache()

    valid_writer.add_scalar('losses/total_loss', valid_loss, epoch_n)
    valid_writer.add_scalar('accuracy/total_accuracy', valid_accuracy, epoch_n)
    valid_writer.add_scalar('train_f1_score/total_train_f1_score', valid_f1_score, epoch_n)
    print('Valid Metrics  on epoch - ' + str(epoch_n))
    print('Accuracy Metrics  on epoch - ' + str(valid_accuracy))
    print('Avg loss  on epoch - ' + str(valid_loss))
    print('F1 - score - ' + str(valid_f1_score))
    print(40*'--')
    print(f'Valid Metrics  on epoch {epoch_n}: \n Accuracy: {(valid_accuracy):>0.3f}, Avg loss: {valid_loss:>8f} \n'
            f'F1 - score: {(valid_f1_score):>0.3f}')

    return valid_loss, valid_accuracy, valid_f1_score