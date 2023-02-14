import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        # self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, virtual_edge_embedding=None):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        self_loop_embeddings = self.edge_embedding1(self_loop_attr[:, 0]) + self.edge_embedding2(self_loop_attr[:, 1])

        if virtual_edge_embedding is not None:
            edge_embeddings = torch.cat([edge_embeddings, virtual_edge_embedding, self_loop_embeddings], dim=0)
        else:
            edge_embeddings = torch.cat([edge_embeddings, self_loop_embeddings], dim=0)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        # self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(aggr)

        # self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__(aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        # self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN_prompt(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN_prompt, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        bottleneck_dim = 30
        self.l1 = 0.1
        self.seq = False
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        self.sequential_prompt = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.sequential_prompt.append(torch.nn.Sequential(
                torch.nn.Linear(emb_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim)
            ))

        self.k = 1
        self.multiple = True
        self.method = 'prefix'

        num_layer_virtual = num_layer if self.multiple else 1
        self.virtual = []
        self.virtual_edge = []
        if self.method == 'qkv':
            self.sim_loss = 0
            self.key = []
        if self.method == 'prefix':
            self.node_mlp = torch.nn.ModuleList()
            self.edge_mlp = torch.nn.ModuleList()

        self.ks = [self.k for _ in range(num_layer_virtual)]
        for i, k in enumerate(self.ks):
            p = nn.Parameter(torch.zeros((k, emb_dim)))
            torch.nn.init.xavier_uniform_(p.data)
            self.virtual.append(p)
            self.register_parameter('virtual_{}'.format(i), self.virtual[-1])

            p_e = nn.Parameter(torch.zeros((k, emb_dim)))
            torch.nn.init.xavier_uniform_(p_e.data)
            self.virtual_edge.append(p_e)
            self.register_parameter('virtual_edge_{}'.format(i), self.virtual_edge[-1])

            if self.method == 'qkv':
                p_k = nn.Parameter(torch.zeros((k, emb_dim)))
                torch.nn.init.xavier_uniform_(p_k.data)
                self.key.append(p_k)
                self.register_parameter('key_{}'.format(i), self.key[-1])

            if self.method == 'prefix':
                self.node_mlp.append(
                    nn.Sequential(
                        nn.Linear(emb_dim, 2 * emb_dim),
                        nn.ReLU(),
                        nn.Linear(2 * emb_dim, emb_dim)
                    )
                )
                self.edge_mlp.append(
                    nn.Sequential(
                        nn.Linear(emb_dim, 2 * emb_dim),
                        nn.ReLU(),
                        nn.Linear(2 * emb_dim, emb_dim)
                    )
                )

        if self.method == 'atte':
            mid_dim = 30
            self.attention_mlp = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * emb_dim, mid_dim),
                               nn.ReLU(),
                               nn.Linear(mid_dim, 1))
                 for p in self.virtual])

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        num_node = x.shape[0]
        extended_edge_index, virtual_edge_embedding = None, None
        if self.method == 'qkv':
            self.sim_loss = 0

        for layer in range(self.num_layer):
            h_ori = h_list[layer]
            if layer == 0 or self.multiple:
                h_list[layer], extended_edge_index, edge_attr, virtual_edge_embedding \
                    = self.merge_virtual(h_list[layer], edge_index, edge_attr, batch, self.virtual[layer],
                                         self.virtual_edge[layer], layer)

            h = self.gnns[layer](h_list[layer], extended_edge_index, edge_attr, virtual_edge_embedding)

            if self.multiple or layer == self.num_layer - 1:
                h = h[:num_node]

            h = self.batch_norms[layer](h)
            if not self.seq:
                h = h * (1 - self.l1) + self.sequential_prompt[layer](h_ori) * self.l1
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            if self.seq:
                h = h * (1 - self.l1) + self.sequential_prompt[layer](h) * self.l1
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

    def mark_largest(self, x, n):
        x = x + torch.rand_like(x) * 1e-4
        sorted_x, _ = torch.sort(x, dim=-1, descending=True)
        top_n = sorted_x[:, n - 1]
        top_n = top_n.unsqueeze(-1).expand(-1, x.shape[-1])
        mask = (x >= top_n)
        if mask.float().sum(dim=1).max() > 1:
            return self.mark_largest(x, n)
        return mask

    def merge_virtual(self, x, edge_index, edge_attr, batch, v, e, layer):
        num_graph = batch.max().item() + 1
        num_v = v.shape[0]
        num_node = x.shape[0]
        emd_dim = v.shape[-1]

        # node
        v_repeat = v.unsqueeze(0).expand(num_graph, -1, -1)
        v_repeat = v_repeat.reshape(-1, emd_dim)

        if self.method == 'full':
            # -----------------------------edge-----------------------------
            virtual_index = batch * num_v
            idx = torch.arange(num_v).unsqueeze(0).expand(num_node, -1).transpose(0, 1).to(x.device)
            virtual_index = idx + virtual_index
            virtual_index = virtual_index.transpose(0, 1) + num_node
            ori_index = torch.arange(num_node).unsqueeze(1).expand(-1, num_v).to(x.device)
            # -----------------------------edge attr-----------------------------
            # fake_attr_0 = torch.arange(num_v) + num_bond_type
            # fake_attr_1 = torch.arange(num_v) + num_bond_direction
            # fake_attr = torch.stack([fake_attr_0, fake_attr_1]).transpose(1, 0)
            # fake_attr = fake_attr.unsqueeze(0).expand(2 * num_node, -1, -1).reshape(-1, 2)
            # fake_attr = fake_attr.to(torch.long).to(x.device)
            idx = torch.arange(num_v).unsqueeze(0).expand(num_node, -1).reshape(-1)
            virtual_edge_embedding = e[idx]
        elif self.method == 'knn':
            # -----------------------------edge-----------------------------
            sim = F.cosine_similarity(x.unsqueeze(1), v, dim=-1)  # batch_size x k
            connect = self.mark_largest(sim, 1)
            idx = torch.where(connect)[1]
            virtual_index = batch * num_v + num_node
            virtual_index = idx + virtual_index.to(x.device)
            ori_index = torch.arange(num_node).to(x.device)
            # -----------------------------edge attr-----------------------------
            # fake_attr = torch.zeros([2]).to(x.device)
            # fake_attr[0], fake_attr[1] = num_bond_type, num_bond_direction
            # fake_attr = fake_attr.unsqueeze(0).expand(num_node, -1)
            # fake_attr = fake_attr.transpose(1, 0) + idx
            # fake_attr = fake_attr.transpose(1, 0)
            # fake_attr = fake_attr.to(torch.long).to(x.device)
            # fake_attr = fake_attr.unsqueeze(0).expand(2, -1, -1).reshape(-1, 2)
            virtual_edge_embedding = e[idx]
        elif self.method == 'qkv':
            # -----------------------------edge-----------------------------
            sim = F.cosine_similarity(x.unsqueeze(1).detach(), self.key[layer], dim=-1)  # batch_size x k
            connect = self.mark_largest(sim, 1)
            self.sim_loss += -sim[connect].mean()
            idx = torch.where(connect)[1]
            virtual_index = batch * num_v + num_node
            virtual_index = idx + virtual_index.to(x.device)
            ori_index = torch.arange(num_node).to(x.device)
            # -----------------------------edge attr-----------------------------
            # fake_attr = torch.zeros([2]).to(x.device)
            # fake_attr[0], fake_attr[1] = num_bond_type, num_bond_direction
            # fake_attr = fake_attr.unsqueeze(0).expand(num_node, -1)
            # fake_attr = fake_attr.transpose(1, 0) + idx
            # fake_attr = fake_attr.transpose(1, 0)
            # fake_attr = fake_attr.to(torch.long).to(x.device)
            # fake_attr = fake_attr.unsqueeze(0).expand(2, -1, -1).reshape(-1, 2)
            virtual_edge_embedding = e[idx]
        elif self.method == 'atte':
            # ----------------------------node---------------------------
            mlp = self.attention_mlp[layer]
            x_r = x.unsqueeze(1).expand(-1, num_v, -1)
            v_r = v.unsqueeze(0).expand(num_node, -1, -1)
            concat = torch.cat([x_r, v_r], dim=-1)  # num_node x num_v x 2*n_feature
            link_score = mlp(concat).squeeze(-1)  # num_node x num_v
            link_score = F.softmax(link_score, dim=-1)  # num_node x num_v
            new_nodes = link_score @ v  # num_node x n_feature
            v_repeat = new_nodes
            # -----------------------------edge-----------------------------
            ori_index = torch.arange(num_node).to(x.device)
            virtual_index = ori_index + num_node
            # -----------------------------edge attr-----------------------------
            # fake_attr = torch.zeros(2).to(x.device)
            # fake_attr[0], fake_attr[1] = num_bond_type, num_bond_direction
            # fake_attr = fake_attr.unsqueeze(0).expand(2 * num_node, -1, -1).reshape(-1, 2)
            # fake_attr = fake_attr.to(torch.long)
            virtual_edge_embedding = link_score @ e
        elif self.method == 'prefix':
            # -----------------------------edge-----------------------------
            virtual_index = batch * num_v
            idx = torch.arange(num_v).unsqueeze(0).expand(num_node, -1).transpose(0, 1).to(x.device)
            virtual_index = idx + virtual_index
            virtual_index = virtual_index.transpose(0, 1) + num_node
            ori_index = torch.arange(num_node).unsqueeze(1).expand(-1, num_v).to(x.device)
            # -----------------------------edge attr-----------------------------
            idx = torch.arange(num_v).unsqueeze(0).expand(num_node, -1).reshape(-1)
            virtual_edge_embedding = e[idx]
            # -----------------------------mlp-----------------------------
            v_repeat = self.node_mlp[layer](v_repeat)
            virtual_edge_embedding = self.edge_mlp[layer](virtual_edge_embedding)

        extra_edge_index_in = torch.stack([ori_index.reshape(-1), virtual_index.reshape(-1)])
        extra_edge_index_out = torch.stack([virtual_index.reshape(-1), ori_index.reshape(-1)])

        # fake_attr = torch.zeros(2).to(x.device)
        # fake_attr[0] = 3
        # fake_attr = fake_attr.unsqueeze(0).expand(2 * extra_edge_index_in.shape[1], -1, -1).reshape(-1, 2)
        # fake_attr = fake_attr.to(torch.long)

        # fake_attr = torch.zeros(2).to(x.device)
        # fake_attr[0], fake_attr[1] = num_bond_type, num_bond_direction
        # fake_attr = fake_attr.unsqueeze(0).expand(2 * extra_edge_index_in.shape[1], -1, -1).reshape(-1, 2)
        # fake_attr = fake_attr.to(torch.long)

        # combine
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        # print(v_repeat.shape)
        # print(extra_edge_index_in.shape)
        # print(fake_attr.shape)
        x = torch.cat([x, v_repeat])
        edge_index = torch.cat([edge_index, extra_edge_index_in, extra_edge_index_out], dim=1)
        virtual_edge_embedding = torch.cat([virtual_edge_embedding, virtual_edge_embedding])
        # edge_attr = torch.cat([edge_attr, fake_attr]).long()
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        # print(extra_edge_index_in.shape)
        # print(extra_edge_index_out.shape)
        # print(virtual_edge_embedding.shape)
        return x, edge_index, edge_attr, virtual_edge_embedding

    # def init_edge_prompt(self):
    #     # init edge
    #     # extend conv edge embedding
    #     num_v = self.k
    #     device = self.gnns[0].edge_embedding1.weight.data.device
    #     for conv in self.gnns:
    #         if self.edge_init == 'mean':
    #             init1 = conv.edge_embedding1.weight.data.mean(dim=0, keepdim=True)
    #             init2 = conv.edge_embedding2.weight.data.mean(dim=0, keepdim=True)
    #         elif self.edge_init == 'xavier':
    #             emd_dim = conv.edge_embedding1.weight.data.shape[-1]
    #             init1 = torch.nn.init.xavier_uniform_(torch.zeros([num_v, emd_dim]))
    #             init2 = torch.nn.init.xavier_uniform_(torch.zeros([num_v, emd_dim]))
    #
    #         conv.edge_embedding1.weight.data = torch.cat([conv.edge_embedding1.weight.data, init1.to(device)])
    #         conv.edge_embedding2.weight.data = torch.cat([conv.edge_embedding2.weight.data, init2.to(device)])

    @torch.no_grad()
    def init_node_prompt(self, train_loader):
        print('init node prompt as mean')
        self.eval()
        num_virtual_layers = self.num_layer if self.multiple else 1
        embeddings = [[] for _ in range(num_virtual_layers)]
        device = self.virtual[0].device
        for data in train_loader:
            data = data.to(device)
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
            embeddings[0].append(h.cpu().detach())
            for layer in range(num_virtual_layers - 1):
                h = self.gnns[layer](h, edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                h = F.relu(h)
                embeddings[layer + 1].append(h.cpu().detach())
        for i, embed in enumerate(embeddings):
            arr = torch.cat(embed, dim=0)
            mean = arr.mean(dim=0)
            self.virtual[i] *= 0
            self.virtual[i] += mean.to(device)
        self.train()

    def load_prompt(self, prompt_file, ori_file=None):
        print('load from initialized prompt')
        self.load_state_dict(torch.load(prompt_file))
        if ori_file is not None:
            self.load_state_dict(torch.load(ori_file), strict=False)


class GNN_prompt_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GNN_prompt_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = 2

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN_prompt(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file), strict=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr, batch)
        node_representation = self.pool(node_representation, batch)

        return node_representation, self.graph_pred_linear(node_representation)


if __name__ == "__main__":
    pass
