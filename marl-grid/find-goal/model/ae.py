from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt #plt 추가했다. obs 출력용

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, ImgModule


class STE(torch.autograd.Function):
    """Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # clamp gradient between -1 and 1
        return F.hardtanh(grad_output)


class InputProcessor(nn.Module):
    """
    Pre-process the following individual observations:
        - pov (ImgModule)
        - self_env_act
        - selfpos
    """
    def __init__(self, obs_space, comm_feat_len, num_agents, last_fc_dim=64):
        super(InputProcessor, self).__init__()

        self.obs_keys = list(obs_space.spaces.keys())
        self.num_agents = num_agents

        # image processor
        assert 'pov' in self.obs_keys
        self.conv = ImgModule(obs_space['pov'].shape, last_fc_dim=last_fc_dim)
        feat_dim = last_fc_dim

        # state inputs processor
        state_feat_dim = 0

        if 'self_env_act' in self.obs_keys:
            # discrete value with one-hot encoding
            self.env_act_dim = obs_space.spaces['self_env_act'].n
            state_feat_dim += self.env_act_dim

        if 'selfpos' in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces['selfpos'].__class__.__name__ == \
                    'MultiDiscrete':
                # process position with one-hot encoder
                self.discrete_positions = obs_space.spaces['selfpos'].nvec
                state_feat_dim += sum(self.discrete_positions)  # 여기서 30이 나온 것이다. 
            else:
                state_feat_dim += 2

        if state_feat_dim == 0:
            self.state_feat_fc = None
        else:
            # use state_feat_fc to process concatenated state inputs
            self.state_feat_fc = nn.Linear(state_feat_dim, 64)
            feat_dim += 64

        if self.state_feat_fc:
            self.state_layer_norm = nn.LayerNorm(64)
        self.img_layer_norm = nn.LayerNorm(last_fc_dim)

        # all other agents' decoded features, if provided
        self.comm_feat_dim = comm_feat_len * 2 * (num_agents - 1)   # comm_feat_len이 128이나 된다고 합니다. 지금 message encoder에 어텐션된 값이랑 기본 값 둘 다 넣기 때문에 곱하기 2를 써줬다.
        feat_dim += self.comm_feat_dim  # feat_dim는 

        self.feat_dim = feat_dim

    def forward(self, inputs, comm=None):   # comm이 없을 땐 이미지 인코더, 있을 땐 두 개를 섞어버리는 인코더가 되는 것 같다.
        # WARNING: the following code only works for Python 3.6 and beyond

        # process images together if provided
        if 'pov' in self.obs_keys:
            pov = []
            for i in range(self.num_agents):
                pov.append(inputs[f'agent_{i}']['pov']) # 각각의 1,3,42,42사이즈의 pov를 수집하고 이를 합쳤다.
                # pov_to_cpuNum = pov[i].to('cpu').detach().numpy()
                # pov_to_img = np.transpose(pov_to_cpuNum[0],(1,2,0))
                # plt.imsave(f'pov_to_img_agent{i}.jpeg', pov_to_img/255.)
            x = torch.cat(pov, dim=0)   #합친 것들을 텐서상으로도 콘캩하는 코드.
            x = self.conv(x)  # (N, img_feat_dim)   3,64의 결과가 나왔다.
            xs = torch.chunk(x, self.num_agents)    # 3개로 또 쪼갠다. 각각 1,64 차원이다.

        # concatenate observation features
        cat_feat = [self.img_layer_norm(xs[i]) for i in range(self.num_agents)]

        if self.state_feat_fc is None:
            if comm is not None:
                for i in range(self.num_agents):
                    # concat comm features for each agent
                    c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                    cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)
            return cat_feat

        for i in range(self.num_agents):
            # concatenate state features
            feats = []

            # concat last env act if provided
            if 'self_env_act' in self.obs_keys: # 이거 보통 false인 것 같다
                env_act = F.one_hot(
                    inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                    num_classes=self.env_act_dim)
                env_act = torch.reshape(env_act, (1, self.env_act_dim))
                feats.append(env_act)

            # concat agent's own position if provided
            if 'selfpos' in self.obs_keys:
                sp = inputs[f'agent_{i}']['selfpos'].to(torch.int64)  # (2,)    # 좌표 포지션이 아웃풋으로 나온다.
                if self.discrete_positions is not None:
                    spx = F.one_hot(sp[0],
                                    num_classes=self.discrete_positions[0]) # self.discrete_positions[0]이 15,15 좌표계에서 좌표계를 꺼내온거고 그 상에서 sp의 x좌표 값을 원핫으로 바꿨다.
                    spy = F.one_hot(sp[1],
                                    num_classes=self.discrete_positions[1])
                    sp = torch.cat([spx, spy], dim=-1).float()
                    sp = torch.reshape(sp, (1, sum(self.discrete_positions))) # sp.shape = [1,30]이다
                else:
                    sp = torch.reshape(sp, (1, 2))
                feats.append(sp)    # feats[0].shape == torch.Size([1,30])

            if len(feats) > 1:
                feats = torch.cat(feats, dim=-1)
            elif len(feats) == 1:
                feats = feats[0]    #feats가 리스트였는데 tensor로 바뀌게 되었다.
            else:
                raise ValueError('?!?!?!', feats)

            feats = self.state_feat_fc(feats)   # 어 이제 1,64 텐서로 바뀌었다.
            feats = self.state_layer_norm(feats)    # [1,64]
            cat_feat[i] = torch.cat([cat_feat[i], feats], dim=-1)   # [1,128]로 바뀐다. pov랑 selfpos랑 합친 것 같다. 신기한게 1,128이 된다는 것?

            if comm is not None:
                # concat comm features for each agent
                # if comm[i].shape[0] == 2:
                c = torch.reshape(comm[i], (1, self.comm_feat_dim)) # 1, 512 으로 reshape 왜냐면 self.comm_feat_dim 자체가 이미 두 배 뻥튀기로 변환되어 있다.
                cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)

        return cat_feat


class EncoderDecoder(nn.Module):
    def __init__(self, obs_space, comm_len, discrete_comm, num_agents,
                 ae_type='', img_feat_dim=64):
        super(EncoderDecoder, self).__init__()

        self.preprocessor = InputProcessor(obs_space, 0, num_agents,
                                           last_fc_dim=img_feat_dim)
        in_size = self.preprocessor.feat_dim    # 여기 있는 인풋프로세서 오브젝트는 feat_dim구하기 위함 ㅋㅋㅋㅋ

        if ae_type == 'rfc':
            # random projection using fc
            self.encoder = nn.Sequential(
                nn.Linear(in_size, comm_len),
                nn.Sigmoid(),
            )
        elif ae_type == 'rmlp':
            # random projection using mlp
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, comm_len),
                nn.Sigmoid()
            )
        elif ae_type == 'fc':
            # fc on AE
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, img_feat_dim),
            )
            self.fc = nn.Sequential(
                nn.Linear(img_feat_dim, comm_len),
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(img_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, in_size),
            )
        elif ae_type == 'mlp':
            # mlp on AE
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, img_feat_dim),
            )
            self.fc = nn.Sequential(
                nn.Linear(img_feat_dim, img_feat_dim),
                nn.ReLU(),
                nn.Linear(img_feat_dim, img_feat_dim),
                nn.ReLU(),
                nn.Linear(img_feat_dim, comm_len),
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(img_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, in_size),
            )
        elif ae_type == '':
            # AE
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, comm_len),
                nn.Sigmoid()
            )
            self.decoder = nn.Sequential(
                nn.Linear(comm_len, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, in_size),    # 128인 것 같다.
            )
        else:
            raise NotImplementedError

        self.discrete_comm = discrete_comm
        self.ae_type = ae_type

    def decode(self, x):
        """
        input: inputs[f'agent_{i}']['comm'] (num_agents, comm_len)
            (note that agent's own state is at the last index)
        """
        if self.ae_type:
            # ['fc', 'mlp', 'rfc', 'rmlp']
            return x
        else:
            return self.decoder(x)  # (num_agents, in_size)

    def forward(self, feat):
        encoded = self.encoder(feat)

        if self.ae_type in {'rfc', 'rmlp'}:
            if self.discrete_comm:
                encoded = STE.apply(encoded)
            return encoded, torch.tensor(0.0)

        elif self.ae_type in {'fc', 'mlp'}:
            decoded = self.decoder(encoded)
            loss = F.mse_loss(decoded, feat)

            comm = self.fc(encoded.detach())
            if self.discrete_comm:
                comm = STE.apply(comm)
            return comm, loss

        elif self.ae_type == '':
            if self.discrete_comm:
                encoded = STE.apply(encoded)
            decoded = self.decoder(encoded)
            loss = F.mse_loss(decoded, feat)
            return encoded.detach(), loss

        else:
            raise NotImplementedError

############################################################
class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_dimension):
        super(MultiHeadAttention, self).__init__()
        self.attention = ScaleDotProductAttention()
        self.W_Q = nn.Linear(model_dimension, model_dimension)
        self.W_K = nn.Linear(model_dimension, model_dimension)
        self.W_V = nn.Linear(model_dimension, model_dimension)
    
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight metrices
        q, k, v = self.W_Q(q), self.W_K(k), self.W_V(v)
        
        # 2. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    
    Query   : given action that we focused on (decoder)
    Key     : an image to check relationship with Query (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None, e=1e-12):
        length, d_tensor = k.size()
        
        # 1. dot product Query with key^T to compute similarity
        k_t = k.transpose(0,1) # transpose
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product
            
        # 2. pass them softmax to make [0,1] range
        score = self.softmax(score)
        
        # 3. multiply with Value
        v = score @ v
        
        return v, score
#################################
class AENetwork(A3CTemplate):
    """
    An network with AE comm.
    """
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type='', hidden_size=256,
                 img_feat_dim=64):
        super().__init__()

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg

        self.num_agents = num_agents

        self.comm_ae = EncoderDecoder(obs_space, comm_len, discrete_comm,
                                      num_agents, ae_type=ae_type,
                                      img_feat_dim=img_feat_dim)    # 여기서 인코더 디코더를 정의를 했다. 나중에 메세지 인코더에서도 이 디코더를 사용한다. 여기서 인코더디코더 객체(내부에 인풋 프로세서 하나 있낀 함) 하나 생성.

        feat_dim = self.comm_ae.preprocessor.feat_dim
        
        ###############################
        # self.multi_head_attention = nn.ModuleList(
        #     [MultiHeadAttention(comm_len, 64
        #               ) for _ in range(num_agents)])
        
        # self.act_feat_fc = nn.ModuleList(
        #     [nn.Linear(self.action_size, comm_len
        #                ) for _ in range(num_agents)])
        
        self.multi_head_attention = MultiHeadAttention(comm_len)
        self.act_feat_fc = nn.Linear(self.action_size, comm_len)
        ################################
        
        if ae_type == '':
            self.input_processor = InputProcessor(
                obs_space,
                feat_dim,   # 아래랑 위랑 두번째 줄만 다르다. 
                num_agents,
                last_fc_dim=img_feat_dim)   # 위에 있는 인코더 디코더랑 다른 그저 인풋프로세서이다. 유념해보자. 여기서 인풋프로세서 객체 하나 생성.
        else:
            self.input_processor = InputProcessor(
                obs_space,
                comm_len,
                num_agents,
                last_fc_dim=img_feat_dim)

        # individual memories
        self.feat_dim = self.input_processor.feat_dim + comm_len
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, num_layers=1
                      ) for _ in range(num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size, 1) for _ in range(num_agents)])
        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.env_action_size) for _ in range(num_agents)])

        self.reset_parameters()
        return

    def reset_parameters(self):
        for m in self.env_actor_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 0.01)
            m.bias.data.fill_(0)

        for m in self.env_critic_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 1.0)
            m.bias.data.fill_(0)
        return

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_action(self, policy_logit, comm_out):  #action policy라고 해야하나
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        all_act_dict = {}
        for agent_name, logits in policy_logit.items():
            act, act_logp, ent = super(AENetwork, self).take_action(logits)

            act_dict[agent_name] = act
            act_logp_dict[agent_name] = act_logp
            ent_list.append(ent)

            comm_act = (comm_out[int(agent_name[-1])]).cpu().numpy()    # 1,10 차원의 comm_out을 그냥 넘파이로 바꾸고 있다.
            all_act_dict[agent_name] = [act, comm_act]
        return act_dict, act_logp_dict, ent_list, all_act_dict

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global

        # WARNING: the following code only works for Python 3.6 and beyond

        # (1) pre-process inputs
        comm_feat = []
        for i in range(self.num_agents):
            #######################
            if inputs[f'agent_{i}'].setdefault('past_action') is not None:
                else_action_list = list(range(self.num_agents))
                del else_action_list[i]
                act_feat = []
                for j in else_action_list:
                    act_hot = F.one_hot(inputs[f'agent_{j}']['past_action'], num_classes=self.action_size)
                    act_feat.append(self.act_feat_fc(act_hot.float())) # act_feat은 1,10의 텐서다.
                    if len(act_feat) == 1:
                        act_feat_tensor = act_feat[0]
                    else:
                        act_feat_tensor = torch.stack((act_feat_tensor, act_feat[-1]), dim=0)
                attention_value = self.multi_head_attention(inputs[f'agent_{i}']['comm'][:-1], inputs[f'agent_{i}']['comm'][:-1], inputs[f'agent_{i}']['comm'][:-1])
                cf_attended = self.comm_ae.decode(attention_value)
                cf_naive = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1])
                cf = torch.cat((cf_attended, cf_naive),dim=0)
            else:
            ##############################
                cf_naive = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1])   # 결국 AENet 생성자 속 오토인코더의 디코더를 사용한게 맞다. 
                cf = torch.cat((cf_naive, cf_naive),dim=0)
            cf = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1]) # 메세지 인코더 돌린 것 같다. 근데 한 줄을 뺐다.
                
            
            if not self.ae_pg:
                cf = cf.detach()
            comm_feat.append(cf)

        cat_feat = self.input_processor(inputs, comm_feat)  # 384까지 뻥튀기가 되기도 한다. 우선 comm_feat는 총 3개 있고, 각각 2,128 차원이다.  #그러니까 이미지 피쳐랑 comm 피쳐 다 합친거다.
                                                            # 생성자에서 생성한 인풋프로세서이며 여기선 comm_feat때문에 아웃풋이 384(+128*2)까지 뻥튀기된다. 

        # (2) generate AE comm output and reconstruction loss
        with torch.no_grad():
            x = self.input_processor(inputs)
        x = torch.cat(x, dim=0)
        comm_out, comm_ae_loss = self.comm_ae(x)    #comm_out이 c_t인 것 같다. x.shape하면 3,128나온다. comm_out은 3,10이다.

        # (3) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}

        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue

            cat_feat[i] = torch.cat([cat_feat[i], comm_out[i].unsqueeze(0)],
                                    dim=-1) # cat_feat[i] == 1,384 comm_out[i].unsqueeze == 1,10

            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i]) #LSTM head에 두 인수를 넣는다.  여기서 hidden_state[:][:].shape == [1,1,256]    cat_feat[i].unsqueeze(0)

            env_actor_out[agent_name] = self.env_actor_linear[i](x)
            env_critic_out[agent_name] = self.env_critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), comm_ae_loss