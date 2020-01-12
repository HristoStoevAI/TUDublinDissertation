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



class EEGAlcoDatasetBalanced():
    """EEG Alco Train dataset."""

    def __init__(self):
        """
        Args:
            none.
        """
        h5f = h5py.File('alco_scalars_balanced_X_train.h5','r')
        self.spikes_seizure_eeg_train = h5f['dataset_alco_scalars_balanced_X_train'][:]
        #self.spikes_seizure_eeg_train=np.swapaxes(self.spikes_seizure_eeg_train,1,2)
#        scalers = {}        
#        for i in range(self.spikes_seizure_eeg.shape[1]):
#            scalers[i] = StandardScaler()
#            self.spikes_seizure_eeg[:, i, :] = scalers[i].fit_transform(self.spikes_seizure_eeg[:, i, :]) 
        h5f.close()
        
        h5f = h5py.File('alco_scalars_balanced_X_test.h5','r')
        self.spikes_seizure_eeg_test = h5f['dataset_alco_scalars_balanced_X_test'][:]
        #self.spikes_seizure_eeg_test=np.swapaxes(self.spikes_seizure_eeg_test,1,2)

        h5f.close()
        
        h5f = h5py.File('alco_balanced_y_train.h5','r')
        self.labels_seizure_eeg_train = h5f['dataset_alco_balanced_y_train'][:]
        h5f.close()

        h5f = h5py.File('alco_balanced_y_test.h5','r')
        self.labels_seizure_eeg_test = h5f['dataset_alco_balanced_y_test'][:]
        h5f.close()

        
    def get_data(self):
        #all folds
        dataArray = list()
            
        trainLabels=self.labels_seizure_eeg_train
        trainValues=self.spikes_seizure_eeg_train
        testLabels=self.labels_seizure_eeg_test
        testValues=self.spikes_seizure_eeg_test

        shuffle = np.random.RandomState(seed=0).permutation(len(trainValues))
        trainValues = trainValues[shuffle]
        trainLabels = trainLabels[shuffle]
        currentSplit = {'X_train': (trainValues), 'X_test': (testValues), 
                        'y_train': (trainLabels), 'y_test': (testLabels)}
        dataArray.append(currentSplit)
        return dataArray

    def __len__(self):
        return len(self.spikes_seizure_eeg_train)

alcoDataset = EEGAlcoDatasetBalanced()
dataArray = alcoDataset.get_data()
X_train = dataArray[0]['X_train']
X_test = dataArray[0]['X_test']
y_train = dataArray[0]['y_train']
y_test = dataArray[0]['y_test']

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



