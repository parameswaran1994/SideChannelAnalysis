import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
from src.utils import perform_attacks, NTGE_fn

train_models = True
dataset = "./"
model_type = "cnn" #mlp, cnn
leakage = "HW" #ID, HW
byte = 0
num_epochs = 50
total_num_models = 100
nb_traces_attacks = 1000


for byte in range(0,16):
    root = "./Result/"
    save_root = root+dataset+"_"+model_type+ "_byte"+str(byte)+"_"+leakage+"/"
    model_root = save_root+"models/"
    print("root:", root)
    print("save_time_path:", save_root)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nb_attacks = 100
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256


    dataloadertrain = Custom_Dataset(root='./Dataset/sysmocom_dataset/', dataset=dataset, leakage=leakage,
                                                 transform=transforms.Compose([ToTensor_trace()]), byte = byte)
    dataloadertrain.choose_phase("train")
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")


    #Random Search

    for num_models in range(total_num_models):

        if train_models == True:
            config = create_hyperparameter_space(model_type)
            np.save(model_root + "model_configuration_"+str(num_models)+".npy", config)
            batch_size = config["batch_size"]
            num_workers = 2
            dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=num_workers),
                           # "test": torch.utils.data.DataLoader(dataloadertest, batch_size=batch_size,
                           #                                     shuffle=True, num_workers=num_workers),
                           "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                              shuffle=True, num_workers=num_workers)
                           }
            dataset_sizes = {"train": len(dataloadertrain), "test": len(dataloadertest), "val": len(dataloaderval)}
            correct_key = dataloadertrain.correct_key
            X_attack = dataloadertrain.X_attack
            Y_attack = dataloadertrain.Y_attack
            plt_attack = dataloadertrain.plt_attack
            num_sample_pts = X_attack.shape[-1]

            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device)
            torch.save(model.state_dict(), model_root + "model_"+str(num_models)+"_byte"+str(byte)+".pth")

        else:
            config = np.load(root, allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            model.load_state_dict(torch.load(model_root + "model_"+str(num_models)+"_byte"+str(byte)+".pth"))

        attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
        predictions_wo_softmax = model(attack_traces)
        predictions = F.softmax(predictions_wo_softmax, dim=1)
        predictions = predictions.cpu().detach().numpy()
        GE, key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, dataset=dataset,
                                             nb_attacks=nb_attacks, shuffle=True, leakage=leakage)

        NTGE = NTGE_fn(GE)
        print("GE", GE)
        print("NTGE", NTGE)
        np.save(model_root + "/result_"+str(num_models)+"_byte"+str(byte), {"GE": GE, "NTGE": NTGE})
    # start = time.time()
    # elapse_time = time.time() - start





