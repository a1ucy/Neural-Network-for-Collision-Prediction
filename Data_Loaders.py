import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
         
        data1 = [i for i in self.data if(i[-1] == 1) and not np.all(i[:5] == 150) and np.all(i[:5] <= 150)]
        data0 = [i for i in self.data if(i[-1] == 0) and np.all(i[:5] <= 150)]
        self.data = np.unique(data1+data0, axis=0)
        
        class_0 = np.array(data0)
        class_1 = np.array(data1)
        if len(class_0) != len(class_1):
            diff = abs(len(class_0) - len(class_1))
            if len(class_0) > len(class_1):
                data_to_prune = class_0
                remain_data = class_1
            else:
                data_to_prune = class_1
                remain_data = class_0
            prune = np.random.choice(data_to_prune.shape[0], diff, replace=False)
            pruned_data = np.delete(data_to_prune, prune, axis = 0)
            self.data = np.vstack((pruned_data, remain_data))
        # print(len(self.data))
        # normalize data and save scaler for inference
        np.savetxt("submission.csv", self.data, delimiter=',')   
        self.scaler = MinMaxScaler()
        self.scaler.clip = False
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms

        
        # np.random.shuffle(self.normalized_data)
        # self.finalized_data = self.normalized_data
        # np.savetxt("submission.csv", self.finalized_data, delimiter=',')    
        # print(len(self.finalized_data))
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
          

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        else:
            x = torch.tensor(self.normalized_data[idx,0:6], dtype=torch.float32)
            y = torch.tensor(self.normalized_data[idx][-1], dtype=torch.float32)
        return {'input': x, 'label': y}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

        # Split the dataset into 80% training and 20% testing
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(self.nav_dataset, [train_size, test_size])
        
        # Create data loaders for training and testing datasets
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
