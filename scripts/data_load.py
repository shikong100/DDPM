import os 
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


Labels = ["ND", "Defect"]

class MultiLabelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(MultiLabelDataset, self).__init__()
        self.annRoot = annRoot
        self.imgRoot = imgRoot
        self.split = split
        self.loader = loader
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        if self.split == "Train":
            self.LabelsName = ["ND"]
            gt = pd.read_csv(gtPath, encoding="utf-8", usecols=["Filename", "ND"])
            imgPaths = []
            for i in range(len(gt["ND"])):
                if gt["ND"][i] == 1:
                    imgPaths.append(gt["Filename"][i])
            self.imgPaths = imgPaths
            
        else:
            self.LabelsName = ["ND", "Defect"]
            gt = pd.read_csv(gtPath, encoding="utf-8", usecols=["Filename", "ND", "Defect"])
            self.imgPaths = gt["Filename"].values
        self.transform = transform
        self.labels = gt[self.LabelsName].values
    
    def __len__(self):
        return (len(self.imgPaths))
    
    def __getitem__(self, index):
        path = self.imgPaths[index]
        img = self.loader(os.path.join(self.imgRoot, self.split, path))

        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        return img, target, path


    

if __name__ == "__main__":
    import sys
    # sys.path.append("../")
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()]
    )

    train = MultiLabelDataset(annRoot="./annotations", imgRoot="/mnt/H/qh_data/Sewer", split="Train", transform=transform)
    img, t, p = train[0]
    print(p)
    data_train = DataLoader(train, batch_size=10, shuffle=False)
    # for data in data_train:
    #     img, t, p = data
    #     print(img)
    #     print(t)
    #     print(p)
    # print(data_train.dataset.__getitem__(0))
    test = MultiLabelDataset(annRoot="./annotations", imgRoot="/mnt/H/qh_data/Sewer", split="Valid", transform=transform)
    data_test = DataLoader(test, batch_size=128, shuffle=False)
    # print(data_test.dataset.__getitem__(0))
