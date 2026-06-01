import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='cora')
        self.parser.add_argument('--source_data', type=str, help="dataset name", default='pubmed')
        self.parser.add_argument('--target_data', type=str, help="dataset name", default='citeseer')
        
        # Model configuration
        self.parser.add_argument('--ckpt', type=str, help="the name of checkpoint", default='pretrained_graphclip')
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.5)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu', 
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        # self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--last_activation', action='store_true', help="the last layer will use activation function or not")
        self.parser.add_argument('--model', type=str, help="model name", default='GNN', 
                                 choices=['GNN'])
        self.parser.add_argument('--norm', type=str, help="the type of normalization, id denotes Identity(w/o norm), bn is batchnorm, ln is layernorm", default='id', 
                                 choices=['id', 'bn', 'ln'])
        # self.parser.add_argument('--encoder', type=str, help="the type of encoder", default='GCN_Encoder', 
        #                          choices=['GCN_Encoder', 'GAT_Encoder', 'SAGE_Encoder', 'GIN_Encoder', 'MLP_Encoder', 'GCNII_Encoder'])
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam', 
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-5)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=1e-5)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=30)
        self.parser.add_argument('--batch_size', type=int, help="the batch size", default=256)
        self.parser.add_argument('--seed', type=int, help="random seed", default=0)
        
        # Processing node attributes
        self.parser.add_argument('--llm', action='store_true', help="use the output of llm as node features")
        self.parser.add_argument('--peft', type=str, help="the type of peft", default='lora', 
                                 choices=['lora', 'prefix', 'prompt', 'adapter', 'ia3'])
        self.parser.add_argument('--lm_type', type=str, help="the type of lm", default='tiny', 
                                 choices=['tiny', 'sbert', 'deberta', 'bert', 'e5', 'llama2', 'llama3', 'llama2-14', 'qwen2', 'qwen2.5-0.5b', 'tiny', 'sbert2'])
        
        # used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=1)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling", default='rw', 
                                 choices=['rw', 'khop', 'shadow'])
    
        # prompt type
        self.parser.add_argument('--prompt', type=str, help="the type of prompt tuning", default='gppt', 
                                 choices=['gppt', 'graphprompt', 'prog', 'gpf'])
        
        #tat
        self.parser.add_argument('--tta', action='store_true', default=False, help='run test-time prompt tuning')
        self.parser.add_argument('--step_size', type=float, default=1e-2, help='step size for perturbation')
        self.parser.add_argument('--add_noise_rate', type=float, default=0.0, help='number of steps for perturbation')
        self.parser.add_argument('--tta_model', type=str, help="the type of tta model", default='tent', 
                                 choices=['tent'])
        # self.parser.add_argument('--lamda', type=float, help="lamda", default='5.0')
        # self.parser.add_argument('--umu', type=float, help="umu", default='1.0')
        
        
        # feature group and mcr2
        self.parser.add_argument('--groups', type=int, default=4,
                            help='feature groups')
        self.parser.add_argument('--gam1', type=float, default=1.,
                    help='gamma1 for tuning empirical loss (default: 1.)')
        self.parser.add_argument('--gam2', type=float, default=1.,
                    help='gamma2 for tuning empirical loss (default: 1.)')
        self.parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared (default: 0.5)')
        self.parser.add_argument('--lamda', type=float, default=1.,
                    help='gamma1 for tuning empirical loss (default: 1.)')
        
         # GNN related
        # self.parser.add_argument("--gnn_model_name", type=str, default='gt')
        # self.parser.add_argument("--gnn_num_layers", type=int, default=4)
        # self.parser.add_argument("--gnn_in_dim", type=int, default=1024)
        # self.parser.add_argument("--gnn_hidden_dim", type=int, default=1024) 
        # self.parser.add_argument("--gnn_num_heads", type=int, default=4)
        # self.parser.add_argument("--gnn_dropout", type=float, default=0.0)
        self.parser.add_argument('--learnable_curvature', action='store_true', help='use learnable curvature, otherwise fixed at -1')
        self.parser.add_argument('--cskg_pretrained', default="hyper_models/ConE_cskg-wncnwd_noscale_reweight0.1_1.10", type=str, help='path for ConE model to load pretrained RotC model')
        self.parser.add_argument('--alpha', type=float, default=0.2, help='alpha for structure-level fusion')
        self.parser.add_argument('--beta', type=float, default=0.2, help='beta for feature-level fusion')
        self.parser.add_argument('--cskg_path', default="datasets/cskg-wncnwd/graph-ConE_cskg-wncnwd_noscale_reweight0.1_1.10.pt", type=str, help='cskg dataset path')
        self.parser.add_argument('--hyper_k', default="3", type=int, help='hyper distance topk')

    def parse_args(self):
        return self.parser.parse_args()