from torch_geometric.data import Batch


def custom_collate(original_batch):
    print("✅ custom_collate 被调用了，batch size =", len(original_batch))
    batch = Batch.from_data_list(original_batch)
    
    
    subgraphs = [ori.sg for ori in original_batch]
    batch_sg = Batch.from_data_list(subgraphs)

    
    batch.sg = batch_sg
    return batch