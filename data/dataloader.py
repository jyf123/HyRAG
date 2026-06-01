from torch.utils.data import DataLoader
from torch_geometric.data import Batch,Data

class NestedGraphLoader(DataLoader):  # ✅ 继承的是 PyTorch 原生 DataLoader
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.nested_collate
        )

    def nested_collate(self, batch_list):
        batch = Batch.from_data_list(batch_list)
        subgraph = [ori.sg for ori in batch_list]
        for data in subgraph:
            # print(type(data))  # 检查是否有非 Data 对象
            if not isinstance(data, Data):
                print(data)
                raise ValueError(f"Invalid type in data_list: {type(data)}")
        batch_sg = Batch.from_data_list(subgraph)
        batch.sg = batch_sg
        return batch

    
    # def custom_collate(self,original_batch):
    #     print("✅ custom_collate 被调用了，batch size =", len(original_batch))
    #     batch = Batch.from_data_list(original_batch)
        
        
    #     subgraphs = [ori.sg for ori in original_batch]
    #     batch_sg = Batch.from_data_list(subgraphs)

        
    #     batch.sg = batch_sg
    #     return batch