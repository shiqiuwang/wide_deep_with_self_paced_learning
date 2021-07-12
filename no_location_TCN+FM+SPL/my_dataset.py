from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        url = self.data[index][:-1]
        label = self.data[index][-1]

        sample = [url, label]

        return sample


# test = MyDataset(test_data)
# test = DataLoader(dataset=test, batch_size=16, shuffle=False)
# for i, data in enumerate(test):
#     print(data[0].shape)
#     print(data[1].shape)
#     print("--------------")
