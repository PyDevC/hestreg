from torch.utils.data import Dataset
from torch import tensor
from torch.types import Tensor

# add for all types of datasets in torch.utils.data.Dataset
# for accessing the types from torch use torch.types
# for transforms type need to check documentation of torchvision.transforms
class ImageDataset(Dataset):
    """base Image dataset class for pytorch
    parameter:
    root: str, This the root folder where the data is stored
    resize: tuple[int], resizes all images from
    transforms: no type yet, uses torchvision.transforms to transform images
    """
    def __init__(self,
                 root:str,
                 resize:tuple[int],
                 transforms:tuple[int],
                 folder_structure:list[str]):
        # in future create a new type called path
        super().__init__()
        self.root = root
        self.labels, self.data = self.load_data()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data)

    def load_data(self)->tuple[list, Tensor]:
        labels = []
        data = []
        data = tensor(data)
        return labels, data
