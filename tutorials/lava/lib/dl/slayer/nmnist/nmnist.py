# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import shutil


import lava.lib.dl.slayer as slayer


def augment(event):
    x_shift = 4
    y_shift = 4
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event


class NMNISTDataset(Dataset):
    """NMNIST dataset method

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default 'data'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default Noney.
    download : bool, optional
        enable/disable automatic download, by default True
    """
    def __init__(
        self, 
        path= os.path.join('data', '468j46mzdv-1'),
        train=True,
        sampling_time=1, sample_length=300,
        transform=None, download=True,
    ):
        super(NMNISTDataset, self).__init__()
        self.path = path
        if train:
            data_path = os.path.join(path, "Train")
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip'
        else:
            data_path = os.path.join(path, "Test")
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip'

        if download is True:
            attribution_text = '''
                NMNIST dataset is freely available here:
                https://www.garrickorchard.com/datasets/n-mnist

                (c) Creative Commons:
                    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
                    "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using
                    Saccades",
                    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015
            '''.replace(' '*12, '')
            if train is True:
                print(attribution_text)

            if not os.path.isdir(data_path):  # dataset does not exist
                os.mkdir(data_path) 
                print(
                    f'NMNIST {"training" if train is True else "testing"} '
                    'dataset is not available locally.'
                )
                print('Attempting download (This will take a while) ...')
                os.makedirs(self.path, exist_ok=True)
                with urllib.request.urlopen(source) as dl_file:
                    with open(f'{self.path}/nmnist.zip', 'wb') as out_file:
                        out_file.write(dl_file.read())
                print('Extracting files ...')
                with zipfile.ZipFile(f'{self.path}', 'r') as zip_ref:
                    zip_ref.extractall(self.path)
                os.remove(f'{self.path}/nmnist.zip')
                print('Download complete.')
        else:
            assert len(glob.glob(f'{data_path}/')) == 0, \
                f'Dataset does not exist. Either set download=True '\
                f'or download it from '\
                f'https://www.garrickorchard.com/datasets/n-mnist '\
                f'to {data_path}/'

        self.samples = glob.glob(f'{data_path}/*/*.bin')
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform

    def __getitem__(self, i):
        filename = self.samples[i]
        label = int(filename.split('\\')[-2])
        event = slayer.io.read_2d_spikes(filename)
        if self.transform is not None:
            event = self.transform(event)
        spike = event.fill_tensor(
                torch.zeros(2, 34, 34, self.num_time_bins),
                sampling_time=self.sampling_time,
            )
        return spike.reshape(-1, self.num_time_bins), label

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : False,
            }
        # neuron_params_drop = {
        #         **neuron_params,
        #         'dropout' : slayer.neuron.Dropout(p=0.05),
        #     }
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(
                    neuron_params_drop, 34*34*2, 512,
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 512, 512,
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 512, 10,
                    weight_norm=True
                ),
            ])

    def forward(self, spike, dt=1.0):

        count = []
        for block in self.blocks:
            print(dt)
            spike = block(spike, dt)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


if __name__ == '__main__':
    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)

    # Sets the device to use for computations to GPU (cuda). This will accelerate training if a compatible GPU is available.

    # device = torch.device('cpu')
    device = torch.device('cuda')

    # Instantiates the SNN (defined in a Network class elsewhere) and moves it to the selected device.
    net = Network().to(device)

    #Uses the Adam optimizer with a learning rate of 0.001 for training the network.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    #Loads the N-MNIST dataset, applying any necessary augmentations to the training set.
    training_set = NMNISTDataset(train=True, transform=augment)
    testing_set = NMNISTDataset(train=False)

    # Wraps the datasets in DataLoader objects to enable batch processing and shuffling.
    train_loader = DataLoader(
            dataset=training_set, batch_size=32, shuffle=True
        )
    test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

    # Defines the loss function using SpikeRate from SLAYER
    error = slayer.loss.SpikeRate(
            true_rate=0.2, false_rate=0.03, reduction='sum'
        ).to(device)
    # error = slayer.loss.SpikeMax(mode='logsoftmax').to(device)

    # keep track of training statistics.
    stats = slayer.utils.LearningStats()

    # Creates an Assistant which handles training, testing, and logging.
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats,
            classifier=slayer.classifier.Rate.predict, count_log=True
        )

    epochs = 200
    dt = 2.0

    for epoch in range(epochs):
        # For each batch in the training data, the Assistant trains the network, computes the output, and logs the event rates (spike counts).
        for i, (input, label) in enumerate(train_loader):  # training loop
            print(str(i) + " it " + str(dt))
            output, count = assistant.train(input, label, dt)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=train_loader)

        # Similar to the training loop but used for testing the network on the test dataset.
        for i, (input, label) in enumerate(test_loader):  # training loop
            output, count = assistant.test(input, label)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=test_loader)

        # Saves the model if the current testing accuracy is the best seen so far.
        # Updates, saves, and plots training statistics.
        # Logs the gradient flow, useful for diagnosing issues with training.
        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
        stats.update()
        stats.save(trained_folder + '/')
        stats.plot(path=trained_folder + '/')
        net.grad_flow(trained_folder + '/')
