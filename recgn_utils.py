# Funcoes auxiliares ao mini projeto Dog Breed Recognition

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from copy import copy
from glob import glob
from PIL import Image, ImageFile
from time import time
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# Checa numero de imagens em cada classe
# Retorna lista de labels e total de imagens
def check_class(data_dir):
    class_names = []
    num_per_class = []
    for d in glob(data_dir + '/*/'):
        class_names.append(d.split('/')[-2])
        num_per_class.append(len(glob(d + '*')))
    df = pd.DataFrame(
        {'Dog_breed': class_names,
         'Images': num_per_class})    
    df = df.sort_values(by=['Images'])
    plt.figure(figsize = (10,40))
    plt.barh(df['Dog_breed'],df['Images'])
    plt.show() 
    return class_names, num_per_class

# Exibe algumas imagens do loader com correspondentes labels
def sample_img_show(loader, class_names, meanm, stdm):
    # Obtem um batch de imagens de treinamento
    dataiter = iter(loader)
    images, labels = dataiter.next()
    images = images.numpy() 
    # Plota imagens com label correspondente
    fig = plt.figure(figsize=(100, 16))
    for idx in np.arange(min(4,len(class_names))):
        ax = fig.add_subplot(4, 20, idx+1, xticks=[], yticks=[])
        image = ((np.transpose(images[idx], (1, 2, 0)) * stdm) + meanm).clip(min=0)
        label = class_names[labels[idx].item()]
        plt.imshow(image)
        plt.title(label)
        
# Executa treinamento do modelo        
def training_step(model, criterion, optimizer, train_loader, epoch, gpu_on):
    # Poe modelo em modo de treinamento
    model.train()
    # Define acumulador
    train_loss = 0.0
    # Percorre os batches...
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Treinando Modelo | Epoca {epoch}')):
        # Move tensores para GPU, se possivel
        if gpu_on:
            data, target = data.cuda(), target.cuda()
        # Limpa gradientes
        optimizer.zero_grad()
        # Calcula saídas, apresentando entradas ao modelo (forward pass)
        output = model(data)
        # Calcula loss do batch
        loss = criterion(output, target)
        # Calcula gradientes do loss (backward pass)
        loss.backward()
        # Realiza passo de otimização, atualizando parametros de treinamento
        optimizer.step()
        # Atualiza loss do treinamento
        train_loss += loss.item()*data.size(0)  
    # Calcula loss medio do treinamento
    train_loss = train_loss/len(train_loader.dataset)
    # Retorna loss medio
    return train_loss   

# Executa validacao do modelo
def validation_step(model, criterion, valid_loader, epoch, gpu_on):
    # Poe modelo em modo de avaliacao
    model.eval()
    # Define acumuladores da validacao
    correct_validation = 0.0
    total_analised = 0.0
    valid_loss = 0.0
    # Percorre os batches...
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader, desc=f'Validando Modelo | Epoca {epoch}')):
        # Move tensores para GPU, se possivel
        if gpu_on:
            data, target = data.cuda(), target.cuda()
        # Calcula saídas, apresentando entradas ao modelo (forward pass)
        output = model(data)
        _, pred = torch.max(output, 1)
        # Compara predicoes corretas e atualiza acumuladores
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not gpu_on else np.squeeze(correct_tensor.cpu().numpy())
        correct_validation += correct.sum()
        total_analised += correct.size
        # Calcula loss do bacth
        loss = criterion(output, target)
        # Atualiza loss da validacao
        valid_loss += loss.item()*data.size(0)
    # Calcula loss medio e acuracia media na validacao    
    valid_loss = valid_loss/len(valid_loader.dataset)
    accuaracy_validation = correct_validation / len(valid_loader.dataset)
    # Retorna loss medio e acuracia media 
    return valid_loss, accuaracy_validation

# Executa treinamento + validacao do modelo
def train_model(model, criterion, optimizer, loaders, num_epochs, gpu_on, max_time = 0):
    start = time()
    # Inicializa acumuladores
    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    valid_accuracy_per_epoch = []
    # Loop de epocas
    for epoch in range(1, num_epochs+1):
        # Realiza fase de treinamento
        train_loss = training_step(model, criterion, optimizer, loaders['train'], epoch, gpu_on)
        # Realiza fase de validacao
        valid_loss, valid_accuracy = validation_step(model, criterion, loaders['valid'], epoch, gpu_on)
        # Atualiza acumuladores
        train_loss_per_epoch.append(train_loss)
        valid_loss_per_epoch.append(valid_loss)
        valid_accuracy_per_epoch.append(valid_accuracy)      
        # Exibe algumas estatisticas de treinamento/validacao
        print(f'Epoca: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tAcuracia: {valid_accuracy*100:.4f}%')
        # Salva modelo a cada epoca
        checkpoint_path = f'model_epoch_{epoch}_acc_{valid_accuracy*100:.4f}_loss_{valid_loss:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'pretrained_model': 'resnet152',
            'criterion': criterion,
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': valid_accuracy,
            'loss': valid_loss
            }, checkpoint_path)
        # Condicoes de parada antecipada:
        # subida da media movel do validation loss
        if (epoch > 10 and (np.mean(valid_loss_per_epoch[-10:]) > np.mean(valid_loss_per_epoch[-11:-1]))): 
            print('Treinamento interrompido: valid loss medio crescente.')
            break
        # tempo maximo de treinamento excedido    
        if (max_time > 0 and (time() - start) > max_time): 
            print('Treinamento interrompido: tempo maximo atingido.')
            break
    # Plota estatisticas de treinamento e validacao
    plt.plot(train_loss_per_epoch, label='Training loss')
    plt.plot(valid_loss_per_epoch, label='Validation loss')
    plt.plot(valid_accuracy_per_epoch, label='Validation Acc')
    plt.legend(frameon=False)
    plt.show()
    return model

# Executa teste do modelo
# Retorna distribuicao de probabilidades nos casos de pass e fail de teste,
# onde a probailidade está no número na saída mais alta da rede após o teste
def test_model(model, criterion, test_loader, gpu_on):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    prob_pass = []
    prob_fail = []
    # Seta modelo em modo de avaliacao
    model.eval()
    # Percorre batches
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc=f'Testando Modelo:')):
        n_batch = batch_idx
        # Move tensores para GPU, se possivel
        if gpu_on:
            data, target = data.cuda(), target.cuda()
        # Calcula saídas, apresentando entradas ao modelo (forward pass)
        output = model(data)
        # Obtem probabilidades
        _, pred = torch.max(torch.exp(output), 1)
        prob = torch.exp(output)
        label = target
        for i in range(len(pred)):
            if pred[i] == label[i]:
                prob_pass.append(prob[i, pred[i]].item())
            else:
                prob_fail.append(prob[i, pred[i]].item())
        # Calcula loss do bacth
        loss = criterion(output, target)
        # Atualiza Test loss medio
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # Converte probabilidade de saida em classe predita
        pred = output.data.max(1, keepdim=True)[1]
        # Compara predicoes corretas e atualiza acumuladores
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('Acuracia do Teste: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))  
    return prob_pass, prob_fail


# Executa teste de unknowns do modelo
# Retorna distribuicao de probabilidades em todos os testes com Unknowns,
# onde a probailidade é o número na saída mais alta da rede após o teste
# Obs: Aqui nao interessa comparativo entre classe predita e classe esperada, mas apenas a probabilidade final
# test_loader: deve conter apenas imagens nao apresentadas a rede em tempo de treinamento
def test_unknowns(test_loader, model, criterion, gpu_on):
    acc_unknown = []
    model.eval()
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc=f'Testando Modelo:')):
        if gpu_on:
            data = data.cuda()
        output = model(data)
        # Probabilidades
        test_features = data
        img = test_features
        _, pred = torch.max(torch.exp(output), 1)
        prob = torch.exp(output)
        for i in range(len(pred)):
            acc_unknown.append(prob[i, pred[i]].item())
    return acc_unknown


# Carrega modelo, classes, dados de normalizacao, e path de imagem
# Retorna classe predita, e sua probabilidade na saida da rede
def predict_breed_dog(model, class_names, img_mean, img_std, img_path, gpu_on):
    # Carrega imagem
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(img_mean, 
                                                         img_std)])
    # 'discard the transparent, 
    # alpha channel (that's the :3) and add the batch dimension'
    image = transform(image)[:3,:,:].unsqueeze(0)
    # Move imagem para GPU se possivel
    if gpu_on:
        image = image.cuda()
    # Calcula label e sua probabilidade
    output = model(image)
    prob, pred = torch.max(torch.exp(output), 1)
    pred_prob = prob.item()
    pred_breed = class_names[pred.item()]
    return pred_breed, pred_prob

# Exibe imagem
def imshow(img_path):
    # load color (BGR) image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.grid(False)
    plt.show()

