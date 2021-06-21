import torch.utils.data
from data.base_data_loader import BaseDataLoader


 # Create data set
def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name())) # The name of the print dataset is ‘AlignedDataset’
    dataset.initialize(opt) # Initialize data set parameter
    return dataset # Return the created dataset

 # Load data set
class CustomDatasetDataLoader(BaseDataLoader): 
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt) # initialization parameters
        self.dataset = CreateDataset(opt) # create data set
        self.dataloader = torch.utils.data.DataLoader( # Load the created data set and customize related parameters
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader # return data set

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size) # return the length of the loaded data set and the maximum load capacity allowed by an epoch
