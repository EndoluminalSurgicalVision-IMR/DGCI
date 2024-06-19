# -*- coding: utf-8 -*-

import torch
from torch.autograd import grad
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import DCI_Modules
import DCI_utils
from pytorch3d.ops.marching_cubes import marching_cubes, marching_cubes_naive
import trimesh



class DCI(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1,
                 hyper_hidden_features=256, hidden_num=128, **kwargs):
        super(DCI, self).__init__()
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        self.immediate_dim = 128
        self.template_code = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=True)
        self.deform_encoder = DCI_Modules.SingleBVPNet(type='sine30', mode='mlp', hidden_features=hidden_num,
                                                       num_hidden_layers=3, in_features=3 + 1,
                                                       out_features=self.immediate_dim,
                                                       outermost_linear=False, last_initial=False)
        self.encoder_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim,
                                              hyper_hidden_layers=hyper_hidden_layers,
                                              hyper_hidden_features=hyper_hidden_features,
                                              hypo_module=self.deform_encoder)

        self.deform_decoder = DCI_Modules.SingleBVPNet(type='sine30', mode='mlp', hidden_features=hidden_num,
                                                       num_hidden_layers=2, in_features=self.immediate_dim,
                                                       out_features=3,
                                                       first_initial=False)

        self.decoder_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim,
                                              hyper_hidden_layers=hyper_hidden_layers,
                                              hyper_hidden_features=hyper_hidden_features,
                                              hypo_module=self.deform_decoder)

        self.sdf_net = DCI_Modules.SDFBVPNet(type='sine30', mode='mlp', hidden_features=hidden_num, num_hidden_layers=4,
                                             in_features=3, out_features=1)
        self.sdf_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=3,
                                          hyper_hidden_features=hyper_hidden_features,
                                          hypo_module=self.sdf_net)

    def forward(self, model_input, gt, epoch_flag=True, **kwargs):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        coords = model_input['coords']
        coords.requires_grad_()
        batchsize = coords.shape[0]
        points_num = coords.shape[1]

        total_embedding = torch.cat([self.template_code.unsqueeze(0), embedding], dim=0)
        sdf_hypo_params = self.sdf_hyper_net(embedding)
        target_sdf_stage1 = self.sdf_net({'coords': coords}, params=sdf_hypo_params)['model_out']

        target_grad_sdf_stage1 = torch.autograd.grad(target_sdf_stage1,
                                                     [coords],
                                                     grad_outputs=torch.ones_like(target_sdf_stage1),
                                                     create_graph=True)[0]

        target_model_in = torch.cat([coords, target_sdf_stage1.detach()], dim=-1)
        encoder_hypo_params = self.encoder_hyper_net(total_embedding)
        target_encoder_hypo_params = {}
        template_encoder_hypo_params = {}
        for k in encoder_hypo_params:
            template_encoder_hypo_params[k] = encoder_hypo_params[k][:1]
            target_encoder_hypo_params[k] = encoder_hypo_params[k][1:]

        target_conditionedbytemp_latent = self.deform_encoder({'coords': target_model_in},
                                                              params=template_encoder_hypo_params)['model_out']

        template_batch_decoder_hypo_params = self.decoder_hyper_net(total_embedding)
        template_decoder_hypo_params: dict = {}
        decoder_hypo_params: dict = {}
        for k, v in template_batch_decoder_hypo_params.items():
            template_decoder_hypo_params[k] = v[:1]
            decoder_hypo_params[k] = v[1:batchsize + 1]

        target2target_output = self.deform_decoder({'coords': target_conditionedbytemp_latent},
                                                   params=decoder_hypo_params)['model_out']  # B,N,3

        target2target_output_sdf = self.sdf_net({'coords': target2target_output},
                                                params=sdf_hypo_params)['model_out']

        zero_latent_code = torch.zeros_like(self.template_code)

        model_out = {
            'model_in': coords,
            'reconstructed_coords': target2target_output,
            'sdf_stage1': target_sdf_stage1,
            'grad_sdf_stage1': target_grad_sdf_stage1,
            'latent_vec': embedding,
            'template_code': self.template_code,
            'target2target_output_sdf': target2target_output_sdf,
            'zero_latent_code': zero_latent_code
        }
        losses = DCI_implicit_loss(model_out, gt)
        return losses

    def detach_sdf(self):
        if not hasattr(self, 'detach_sdf_hyper_net'):
            self.detach_sdf_hyper_net = DCI_utils.detach_new_module(self.sdf_hyper_net)
        else:
            print('already')

    def get_latent_code(self, instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    def inference(self, coords, embedding):
        with torch.no_grad():
            additional_embedding = self.template_code[None, :]
            sdf_hypo_params = self.sdf_hyper_net(embedding)
            sdf_stage = self.sdf_net({'coords': coords}, params=sdf_hypo_params)['model_out']
            if self.template_code.allclose(embedding[0]):
                return sdf_stage1
            encoder_hypo_params = self.encoder_hyper_net(additional_embedding)
            model_in = torch.cat([coords, sdf_stage1], dim=-1)
            latent_feature = self.deform_encoder({'coords': model_in}, params=encoder_hypo_params)['model_out']  # B,N,D
            decoder_hypo_params = self.decoder_hyper_net(embedding)

            output = self.deform_decoder({'coords': latent_feature}, params=decoder_hypo_params)['model_out']  # B,N,3
            new_coords = output[..., :3]
            sdf_hypo_params = self.sdf_hyper_net(embedding)
            sdf_final = self.sdf_net({'coords': new_coords},
                                     params=sdf_hypo_params)['model_out']
        return sdf_final
