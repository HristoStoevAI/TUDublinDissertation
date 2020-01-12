import numpy as np
import h5py
np.random.seed(1000)

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
n_splits=4
train_size=0.8
test_size=0.2

eeg2 = r""



class EEGSchizoDatasetBalanced():
    """EEG Alco Train dataset."""

    def __init__(self):
        """
        Args:
            none.
        """
        h5f = h5py.File('schizo_scalars_unbalanced.h5','r')
        self.spikes_seizure_eeg = h5f['dataset_schizo_scalars_unbalanced'][:]
        self.spikes_seizure_eeg=np.swapaxes(self.spikes_seizure_eeg,1,2)
        scalers = {}        
        for i in range(self.spikes_seizure_eeg.shape[1]):
            scalers[i] = StandardScaler()
            self.spikes_seizure_eeg[:, i, :] = scalers[i].fit_transform(self.spikes_seizure_eeg[:, i, :]) 


        h5f.close()
        
        h5f = h5py.File('schizo_labels_unbalanced.h5','r')
        self.labels_seizure_eeg = h5f['dataset_schizo_labels_unbalanced'][:]
        print(str(np.sum(self.labels_seizure_eeg))+'/'+str(len(self.labels_seizure_eeg)))
        h5f.close()

        
    def get_data(self):
        #all folds
        dataArray = list()
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=0)
        for train_index, test_index in sss.split(self.spikes_seizure_eeg, self.labels_seizure_eeg):
            
            
            trainLabels=self.labels_seizure_eeg[train_index]
            trainValues=self.spikes_seizure_eeg[train_index]
            testLabels=self.labels_seizure_eeg[test_index]
            testValues=self.spikes_seizure_eeg[test_index]

            #BALANCING TRAINING DATA
            positivesIndices=trainLabels==1
            positiveEEGs=trainValues[positivesIndices]
            negativeEEGs=trainValues[~positivesIndices]
            print('positiveEEGs: '+str(len(positiveEEGs)))
            print('negativeEEGs: '+str(len(negativeEEGs)))

            n=np.min([len(positiveEEGs),len(negativeEEGs)])
            print(n)

            trainValues=(np.concatenate((positiveEEGs[0:n],negativeEEGs[0:n]),axis=0))
            trainLabels=(np.concatenate((np.full((n),1),np.full((n),0)),axis=0))
            
            shuffle = np.random.RandomState(seed=0).permutation(len(trainValues))
            trainValues = trainValues[shuffle]
            trainLabels = trainLabels[shuffle]
            currentSplit = {'X_train': (trainValues), 'X_test': (testValues), 
                            'y_train': (trainLabels), 'y_test': (testLabels)}
            dataArray.append(currentSplit)
        return dataArray

    

    def __len__(self):
        return len(self.spikes_seizure_eeg)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eeg = torch.tensor(self.spikes_seizure_eeg[idx])
        print('eeg size (in getitem): '+str(eeg.size()))
        label = self.labels_seizure_eeg[idx]
            
        sample = {'eeg': eeg, 'label': label}
        return sample

schizoDataset = EEGSchizoDatasetBalanced()
dataArray = schizoDataset.get_data()
X_train = dataArray[2]['X_train']
X_test = dataArray[2]['X_test']
y_train = dataArray[2]['y_train']
y_test = dataArray[2]['y_test']

print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

print('TESTING')


np.save(eeg2 + 'X_train.npy', X_train)
np.save(eeg2 + 'y_train.npy', y_train)
np.save(eeg2 + 'X_test.npy', X_test)
np.save(eeg2 + 'y_test.npy', y_test)


