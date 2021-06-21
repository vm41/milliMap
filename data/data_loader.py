
# Create data set loading main function
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name()) # The name returned is "CustomDatasetDataLoader"
    data_loader.initialize(opt) # initialization parameters
    return data_loader
