### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, time
import numpy as np
from torch import optim

from Network import ResNet
from ImageUtils import parse_record
import torch.nn as nn
import Network
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = Network.MyNetwork()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(device)
        self.wd = 5e-4
        # self.network = ResNet(configs)

    def model_setup(self,configs):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=configs["learning_rate"], weight_decay=self.wd, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        # pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        self.model_setup(configs)
        num_batches = num_samples // configs["batch_size"]

        print('### Training... ###')
        for epoch in range(1, configs["max_epoch"] + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            for i in range(num_batches):
                x_train_batch = np.zeros([configs["batch_size"], 3, 32, 32])
                y_train_batch = np.zeros([configs["batch_size"]])
                for single_i in range(configs["batch_size"]):
                    x_train_single = curr_x_train[i * configs["batch_size"] + single_i, :]
                    y_train_single = curr_y_train[i * configs["batch_size"] + single_i]
                    x_train_batch[single_i] = np.transpose(x_train_single, (2,0,1))
                    y_train_batch[single_i] = y_train_single
                x_train_batch_cuda = torch.tensor(x_train_batch).type(torch.FloatTensor).cuda()
                y_train_batch_cuda = torch.tensor(y_train_batch).type(torch.LongTensor).cuda()

                self.optimizer.zero_grad()
                outputs_prob = self.network(x_train_batch_cuda)
                loss = self.criterion(outputs_prob, y_train_batch_cuda)

                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % configs["save_interval"] == 0:
                self.save(epoch,configs)
            self.scheduler.step()
    def save(self, epoch,configs):
        checkpoint_path = os.path.join(configs["modeldir"], 'model-%d.ckpt' % (epoch))
        os.makedirs(configs["modeldir"], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def evaluate(self, x, y, checkpoint_num_list,configs):

        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(configs["modeldir"], 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)
            preds = []
            for i in tqdm(range(x.shape[0])):
                x_train = np.transpose(x[i], (2,0,1))
                x_train = torch.tensor(x_train).type(torch.FloatTensor).unsqueeze(0).to('cuda:0')
                logits = self.network(x_train).to('cuda:0')
                pred = logits.argmax(dim=1)
                preds.append(pred)

            y = torch.tensor(y).clone()
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

    def predict_prob(self, x, checkpoint_num_list, configs):
        self.network.eval()
        print('### Prediction ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(configs["modeldir"], 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)
            preds = np.zeros((x.shape[0],10))
            for i in tqdm(range(x.shape[0])):
                # x_train = np.transpose(x[i], (2, 0, 1))
                # x_train = x[i]
                x_train = x[i].reshape((3, 32, 32))
                # x_train = np.transpose(x_train, [1, 2, 0])
                x_train = torch.tensor(x_train).type(torch.FloatTensor).unsqueeze(0).to('cuda:0')
                logits = self.network(x_train).to('cuda:0')
                pred = logits.cpu().data.numpy()
                preds[i,:] = pred

        return preds

### END CODE HERE