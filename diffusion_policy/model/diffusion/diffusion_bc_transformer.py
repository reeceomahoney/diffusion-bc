from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class DiffusionBCTransformer(ModuleAttrMixin):
    def __init__(self,
            obs_dim: int,
            act_dim: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            n_trans_emb: int,
            p_drop: float=0.0,
        ) -> None:
        super().__init__()

        # embedding
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_trans_emb),
        )
        self.act_emb = nn.Sequential(
            nn.Linear(act_dim, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_trans_emb),
        )
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_trans_emb),
        )
        self.pos_emb = SinusoidalPosEmb(n_trans_emb)
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_trans_emb,
            nhead=n_head,
            dim_feedforward=4*n_trans_emb,
            dropout=p_drop,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer
        )

        # decoder
        self.ln_f = nn.LayerNorm(4*n_trans_emb)
        self.head = nn.Linear(4*n_trans_emb, act_dim)

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.LeakyReLU,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Sequential):
                for submodule in module:
                    submodule.apply(self._init_weights)
        elif isinstance(module, DiffusionBCTransformer):
            pass
        #     torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        #     if module.cond_obs_emb is not None:
        #         torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        # if self.cond_pos_emb is not None:
        #     no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, obs: torch.Tensor, 
                act: torch.Tensor, 
                timestep: torch.Tensor) -> torch.Tensor:
        """
        obs: (B,T_obs,obs_dim)
        act: (B,act_dim)
        timestep: (B,) or int, diffusion step
        output: (B,act_dim)
        """
        # time
        if len(timestep.shape) < 2:
            timestep = timestep.expand(obs.shape[0]).to(obs.device)
        time_emb = self.time_emb(timestep).unsqueeze(1)
        # (B,1,n_trans_emb)

        # embedding
        act_emb = self.act_emb(act)
        obs_emb = [self.obs_emb(obs).unsqueeze(1) for obs in torch.unbind(obs, dim=1)]
        embs = [time_emb, act_emb, *obs_emb]

        # positional embedding
        for idx, e in enumerate(embs):
            e += self.pos_emb(torch.zeros(obs.shape[0], 1).to(obs.device) + idx)

        input_emb = torch.cat(embs, dim=1)
        # (B,4,n_trans_emb)

        x = self.encoder(input_emb)
        x = x.reshape(obs.shape[0], -1)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x.unsqueeze(1)

def test():
    model = DiffusionBCTransformer(
        obs_dim=33,
        act_dim=12,
        n_layer=4,
        n_head=16,
        n_emb=128,
        n_trans_emb=1024,
        p_drop=0.0,
    )

    obs = torch.randn(2, 2, 33)
    act = torch.randn(2, 1, 12)
    timestep = torch.randint(0, 50, (2,))
    out = model(obs, act, timestep)

    timestep = torch.randint(0, 50, (1,))
    out = model(obs, act, timestep)
    print(out.shape)


if __name__ == "__main__":
    test()