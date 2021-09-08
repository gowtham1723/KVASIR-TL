import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import statistics
from scipy.stats import sem
import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm
from model import *
from pylab import savefig

# Data augmentation and normalization for training
# Just normalization for validation & test
data_transforms = {
    'train1': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'test1': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'train2': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'test2': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'train3': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'test3': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'train4': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'test4': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'train5': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'test5': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

}

data_dir = 'data1'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in
                  ['train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4', 'train5', 'test5']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                              shuffle=True, num_workers=0)
               for x in ['train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4', 'train5', 'test5']}
dataset_sizes = {x: len(image_datasets[x]) for x in
                 ['train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4', 'train5', 'test5']}
class_names = image_datasets['train1'].classes

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

#Parameters
num_classes = 8
num_epochs=20
stepsize=20
total_images=800
dataset_name="v1_8class"
name_of_model=dataset_name+"_resnet_custom_SGD_"+str(num_epochs)+"_momentum0point9_with_"+str(stepsize)+"stepsize"
results="Results_directory.txt"

# Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs, fold,name):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train'+fold, 'test'+fold]:
            if phase == 'train'+fold:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'+fold):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train'+fold:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # For graph generation
            if phase == "train"+fold:
                train_loss.append(running_loss / dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "test"+fold:
                val_loss.append(running_loss / dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # for printing
            if phase == "train"+fold:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "test"+fold:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'test'+fold and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model,name_of_model+"fold "+fold+" .pth")
    # Plot the train & validation losses
    plt.figure(1)
    plt.title(fold+"-fold Training Vs Validation Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_counter_train[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], train_loss[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], color='r', label="Training Loss")
    plt.plot(epoch_counter_val[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], val_loss[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], color='g', label="Validation Loss")
    plt.legend()
    plt.savefig(fold+"_Losses_"+name_of_model+ ".png")
    plt.close(1)

    # Plot the accuracies in train & validation
    plt.figure(2)
    plt.title(fold+"-fold Training Vs Validation Accuracies")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epoch_counter_train[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], train_acc[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], color='r', label="Training Accuracy")
    plt.plot(epoch_counter_val[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], val_acc[(num_epochs*(int(fold)-1)):(num_epochs*(int(fold)))], color='g', label="Validation Accuracy")
    plt.legend()
    plt.savefig(fold+"_Accuracies_"+name_of_model+".png")
    plt.close(2)
    return model

def eval_model(model,fold):
    confusion_matrix = cm.ConfusionMeter(8)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test'+fold]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            confusion_matrix.add(predicted, labels)
        print(confusion_matrix.conf)
        # Confusion matrix as a heatmap
        con_m = confusion_matrix.conf
        text_file = open(results, "a")
        text_file.write(str(con_m) + "\n")
        text_file.close()
        return confusion_matrix






model_ft = DenseNet(8)
model_ft.to(device)
print(model_ft)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=stepsize, gamma=0.1)

text_file = open(results, "a")
text_file.write("-----------------------------------\n")
text_file.write(name_of_model+"\n")
text_file.close()

model_ft1 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,fold="1",name=name_of_model)

model_ft = DenseNet(8)
model_ft.to(device)
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=stepsize, gamma=0.1)
model_ft2 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,fold="2",name=name_of_model)

model_ft = DenseNet(8)
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=stepsize, gamma=0.1)
model_ft3 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,fold="3",name=name_of_model)

model_ft = DenseNet(8)
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=stepsize, gamma=0.1)
model_ft4 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,fold="4",name=name_of_model)

model_ft = DenseNet(8)
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=stepsize, gamma=0.1)
model_ft5 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,fold="5",name=name_of_model)


confusion_matrix = eval_model(model_ft1,fold="1")
confusion_matrix2 = eval_model(model_ft2,fold="2")
confusion_matrix3 = eval_model(model_ft3,fold="3")
confusion_matrix4 = eval_model(model_ft4,fold="4")
confusion_matrix5 = eval_model(model_ft5,fold="5")

con = confusion_matrix.conf
con1 = confusion_matrix2.conf
con2 = confusion_matrix3.conf
con3 = confusion_matrix4.conf
con4 = confusion_matrix5.conf

TP = np.zeros(8)
FN = np.zeros(8)
TN = np.zeros(8)
FP = np.zeros(8)
precision = np.zeros(8)
recall = np.zeros(8)
accuracy = np.zeros(8)
specificity = np.zeros(8)
F1_score = np.zeros(8)
MCC = np.zeros(8)
TP1 = np.zeros(8)
FN1 = np.zeros(8)
TN1 = np.zeros(8)
FP1 = np.zeros(8)
precision1 = np.zeros(8)
recall1 = np.zeros(8)
accuracy1 = np.zeros(8)
specificity1 = np.zeros(8)
F1_score1 = np.zeros(8)
MCC1 = np.zeros(8)
TP2 = np.zeros(8)
FN2 = np.zeros(8)
TN2 = np.zeros(8)
FP2 = np.zeros(8)
precision2 = np.zeros(8)
recall2 = np.zeros(8)
accuracy2 = np.zeros(8)
specificity2 = np.zeros(8)
F1_score2 = np.zeros(8)
MCC2 = np.zeros(8)
TP3 = np.zeros(8)
FN3 = np.zeros(8)
TN3 = np.zeros(8)
FP3 = np.zeros(8)
precision3 = np.zeros(8)
recall3 = np.zeros(8)
accuracy3 = np.zeros(8)
specificity3 = np.zeros(8)
F1_score3 = np.zeros(8)
MCC3 = np.zeros(8)
TP4 = np.zeros(8)
FN4 = np.zeros(8)
TN4 = np.zeros(8)
FP4 = np.zeros(8)
precision4 = np.zeros(8)
recall4 = np.zeros(8)
accuracy4 = np.zeros(8)
specificity4 = np.zeros(8)
F1_score4 = np.zeros(8)
MCC4 = np.zeros(8)

for i in range(8):
    TP[i] = con[i, i]
    TP1[i] = con1[i, i]
    TP2[i] = con2[i, i]
    TP3[i] = con3[i, i]
    TP4[i] = con4[i, i]
for i in range(8):
    a = 0
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for j in range(8):
        if i != j:
            a = a + con[i, j]
            a1 = a1 + con1[i, j]
            a2 = a2 + con2[i, j]
            a3 = a3 + con3[i, j]
            a4 = a4 + con4[i, j]
        else:
            a = a
            a1 = a1
            a2 = a2
            a3 = a3
            a4 = a4

    FN[i] = a
    FN1[i] = a1
    FN2[i] = a2
    FN3[i] = a3
    FN4[i] = a4

for i in range(8):
    a = 0
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for j in range(8):
        if i != j:
            a = a + con[j, j]
            a1 = a1 + con1[j, j]
            a2 = a2 + con2[j, j]
            a3 = a3 + con3[j, j]
            a4 = a4 + con4[j, j]
        else:
            a = a
            a1 = a1
            a2 = a2
            a3 = a3
            a4 = a4

    TN[i] = a
    TN1[i] = a1
    TN2[i] = a2
    TN3[i] = a3
    TN4[i] = a4

for i in range(8):
    a = 0
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for j in range(8):
        if i != j:
            a = a + con[j, i]
            a1 = a1 + con1[j, i]
            a2 = a2 + con2[j, i]
            a3 = a3 + con3[j, i]
            a4 = a4 + con4[j, i]
        else:
            a = a
            a1 = a1
            a2 = a2
            a3 = a3
            a4 = a4
    FP[i] = a
    FP1[i] = a1
    FP2[i] = a2
    FP3[i] = a3
    FP4[i] = a4

for i in range(8):
    TN[i] = total_images - (TP[i] + FN[i] + FP[i])
    TN1[i] = total_images - (TP1[i] + FN1[i] + FP1[i])
    TN2[i] = total_images - (TP2[i] + FN2[i] + FP2[i])
    TN3[i] = total_images - (TP3[i] + FN3[i] + FP3[i])
    TN4[i] = total_images - (TP4[i] + FN4[i] + FP4[i])

print("True Positives: ", TP)
print("False Negatives: ", FN)
print("True Negatives: ", TN)
print("False Positives", FP)
print("True Positives1: ", TP1)
print("False Negatives1: ", FN1)
print("True Negatives1: ", TN1)
print("False Positives1:", FP1)
av_accuracy = 0
av_precision = 0
av_recall = 0
av_specificity = 0
av_F1_score = 0
av_MCC = 0
av_accuracy1 = 0
av_precision1 = 0
av_recall1 = 0
av_specificity1 = 0
av_F1_score1 = 0
av_MCC1 = 0
av_accuracy2 = 0
av_precision2 = 0
av_recall2 = 0
av_specificity2 = 0
av_F1_score2 = 0
av_MCC2 = 0
av_accuracy3 = 0
av_precision3 = 0
av_recall3 = 0
av_specificity3 = 0
av_F1_score3 = 0
av_MCC3 = 0
av_accuracy4 = 0
av_precision4 = 0
av_recall4 = 0
av_specificity4 = 0
av_F1_score4 = 0
av_MCC4 = 0

for i in range(8):
    precision[i] = (TP[i] / (TP[i] + FP[i]))
    recall[i] = (TP[i] / (TP[i] + FN[i]))
    accuracy[i] = ((TP[i] + TN[i]) / (total_images))
    specificity[i] = ((TN[i]) / (FP[i] + TN[i]))
    F1_score[i] = ((2 * TP[i]) / ((2 * TP[i]) + FP[i] + FN[i]))
    MCC[i] = (((TP[i] * TN[i]) - (FP[i] * FN[i])) / (
        math.sqrt((TP[i] + FP[i]) * (TP[i] + FN[i]) * (TN[i] + FP[i]) * (TN[i] + FN[i]))))
    av_accuracy = av_accuracy + accuracy[i]
    av_precision = av_precision + precision[i]
    av_recall = av_recall + recall[i]
    av_specificity = av_specificity + specificity[i]
    av_F1_score = av_F1_score + F1_score[i]
    av_MCC = av_MCC + MCC[i]

    precision1[i] = (TP1[i] / (TP1[i] + FP1[i]))
    recall1[i] = (TP1[i] / (TP1[i] + FN1[i]))
    accuracy1[i] = ((TP1[i] + TN1[i]) / (total_images))
    specificity1[i] = ((TN1[i]) / (FP1[i] + TN1[i]))
    F1_score1[i] = ((2 * TP1[i]) / ((2 * TP1[i]) + FP1[i] + FN1[i]))
    MCC1[i] = (((TP1[i] * TN1[i]) - (FP1[i] * FN1[i])) / (
        math.sqrt((TP1[i] + FP1[i]) * (TP1[i] + FN1[i]) * (TN1[i] + FP1[i]) * (TN1[i] + FN1[i]))))
    av_accuracy1 = av_accuracy1 + accuracy1[i]
    av_precision1 = av_precision1 + precision1[i]
    av_recall1 = av_recall1 + recall1[i]
    av_specificity1 = av_specificity1 + specificity1[i]
    av_F1_score1 = av_F1_score1 + F1_score1[i]
    av_MCC1 = av_MCC1 + MCC1[i]

    precision2[i] = (TP2[i] / (TP2[i] + FP2[i]))
    recall2[i] = (TP2[i] / (TP2[i] + FN2[i]))
    accuracy2[i] = ((TP2[i] + TN2[i]) / (total_images))
    specificity2[i] = ((TN2[i]) / (FP2[i] + TN2[i]))
    F1_score2[i] = ((2 * TP2[i]) / ((2 * TP2[i]) + FP2[i] + FN2[i]))
    MCC2[i] = (((TP2[i] * TN2[i]) - (FP2[i] * FN2[i])) / (
        math.sqrt((TP2[i] + FP2[i]) * (TP2[i] + FN2[i]) * (TN2[i] + FP2[i]) * (TN2[i] + FN2[i]))))
    av_accuracy2 = av_accuracy2 + accuracy2[i]
    av_precision2 = av_precision2 + precision2[i]
    av_recall2 = av_recall2 + recall2[i]
    av_specificity2 = av_specificity2 + specificity2[i]
    av_F1_score2 = av_F1_score2 + F1_score2[i]
    av_MCC2 = av_MCC2 + MCC2[i]

    precision3[i] = (TP3[i] / (TP3[i] + FP3[i]))
    recall3[i] = (TP3[i] / (TP3[i] + FN3[i]))
    accuracy3[i] = ((TP3[i] + TN3[i]) / (total_images))
    specificity3[i] = ((TN3[i]) / (FP3[i] + TN3[i]))
    F1_score3[i] = ((2 * TP3[i]) / ((2 * TP3[i]) + FP3[i] + FN3[i]))
    MCC3[i] = (((TP3[i] * TN3[i]) - (FP3[i] * FN3[i])) / (
        math.sqrt((TP3[i] + FP3[i]) * (TP3[i] + FN3[i]) * (TN3[i] + FP3[i]) * (TN3[i] + FN3[i]))))
    av_accuracy3 = av_accuracy3 + accuracy3[i]
    av_precision3 = av_precision3 + precision3[i]
    av_recall3 = av_recall3 + recall3[i]
    av_specificity3 = av_specificity3 + specificity3[i]
    av_F1_score3 = av_F1_score3 + F1_score3[i]
    av_MCC3 = av_MCC3 + MCC3[i]

    precision4[i] = (TP4[i] / (TP4[i] + FP4[i]))
    recall4[i] = (TP4[i] / (TP4[i] + FN4[i]))
    accuracy4[i] = ((TP4[i] + TN4[i]) / (total_images))
    specificity4[i] = ((TN4[i]) / (FP4[i] + TN4[i]))
    F1_score4[i] = ((2 * TP4[i]) / ((2 * TP4[i]) + FP4[i] + FN4[i]))
    MCC4[i] = (((TP4[i] * TN4[i]) - (FP4[i] * FN4[i])) / (
        math.sqrt((TP4[i] + FP4[i]) * (TP4[i] + FN4[i]) * (TN4[i] + FP4[i]) * (TN4[i] + FN4[i]))))
    av_accuracy4 = av_accuracy4 + accuracy4[i]
    av_precision4 = av_precision4 + precision4[i]
    av_recall4 = av_recall4 + recall4[i]
    av_specificity4 = av_specificity4 + specificity4[i]
    av_F1_score4 = av_F1_score4 + F1_score4[i]
    av_MCC4 = av_MCC4 + MCC4[i]

av_accuracy = av_accuracy / 8
av_precision = av_precision / 8
av_recall = av_recall / 8
av_specificity = av_specificity / 8
av_F1_score = av_F1_score / 8
av_MCC = av_MCC / 8

av_accuracy1 = av_accuracy1 / 8
av_precision1 = av_precision1 / 8
av_recall1 = av_recall1 / 8
av_specificity1 = av_specificity1 / 8
av_F1_score1 = av_F1_score1 / 8
av_MCC1 = av_MCC1 / 8

av_accuracy2 = av_accuracy2 / 8
av_precision2 = av_precision2 / 8
av_recall2 = av_recall2 / 8
av_specificity2 = av_specificity2 / 8
av_F1_score2 = av_F1_score2 / 8
av_MCC2 = av_MCC2 / 8

av_accuracy3 = av_accuracy3 / 8
av_precision3 = av_precision3 / 8
av_recall3 = av_recall3 / 8
av_specificity3 = av_specificity3 / 8
av_F1_score3 = av_F1_score3 / 8
av_MCC3 = av_MCC3 / 8

av_accuracy4 = av_accuracy4 / 8
av_precision4 = av_precision4 / 8
av_recall4 = av_recall4 / 8
av_specificity4 = av_specificity4 / 8
av_F1_score4 = av_F1_score4 / 8
av_MCC4 = av_MCC4 / 8

accuracy_folds = [av_accuracy, av_accuracy1, av_accuracy2, av_accuracy3, av_accuracy4]
precision_folds = [av_precision, av_precision1, av_precision2, av_precision3, av_precision4]
recall_folds = [av_recall, av_recall1, av_recall2, av_recall3, av_recall4]
specificity_folds = [av_specificity, av_specificity1, av_specificity2, av_specificity3, av_specificity4]
F1_score_folds = [av_F1_score, av_F1_score1, av_F1_score2, av_F1_score3, av_F1_score4]
MCC_folds = [av_MCC, av_MCC1, av_MCC2, av_MCC3, av_MCC4]

overall_accuracy = statistics.mean(accuracy_folds)
overall_precision = statistics.mean(precision_folds)
overall_recall = statistics.mean(recall_folds)
overall_specificity = statistics.mean(specificity_folds)
overall_F1_score = statistics.mean(F1_score_folds)
overall_MCC = statistics.mean(MCC_folds)

sem_a = sem(accuracy_folds)
sem_p = sem(precision_folds)
sem_r = sem(recall_folds)
sem_s = sem(specificity_folds)
sem_f = sem(F1_score_folds)
sem_m = sem(MCC_folds)

print("OVERALL Precision: " + str(overall_precision))
print("OVERALL recall: " + str(overall_recall))
print("OVERALL specificty: " + str(overall_specificity))
print("OVERALL accuracy: " + str(overall_accuracy))
print("OVERALL MCC: " + str(overall_MCC))
print("OVERALL F1_score: " + str(overall_F1_score))
text_file = open(results, "a")
text_file.write("precision_folds" + str(precision_folds) + "\n")
text_file.write("recall_folds" + str(recall_folds) + "\n")
text_file.write("specificity_folds" + str(specificity_folds) + "\n")
text_file.write("accuracy_folds" + str(accuracy_folds) + "\n")
text_file.write("MCC_folds" + str(MCC_folds) + "\n")
text_file.write("F1_score_folds" + str(F1_score_folds) + "\n")
text_file.write("OVERALL Precision: " + str(overall_precision) + "+-" + str(sem_p) + "\n")
text_file.write("OVERALL recall: " + str(overall_recall) + "+-" + str(sem_r) + "\n")
text_file.write("OVERALL specificity: " + str(overall_specificity) + "+-" + str(sem_s) + "\n")
text_file.write("OVERALL accuracy: " + str(overall_accuracy) + "+-" + str(sem_a) + "\n")
text_file.write("OVERALL MCC: " + str(overall_MCC) + "+-" + str(sem_m) + "\n")
text_file.write("OVERALL F1_score: " + str(overall_F1_score) + "+-" + str(sem_f) + "\n")
text_file.write("-----------------------------------\n")
text_file.close()
