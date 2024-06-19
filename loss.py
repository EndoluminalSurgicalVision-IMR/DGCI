# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F

import numpy as np
import DCI_utils


def DCI_implicit_loss(model_output, gt):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    reconstructed_coords = model_output['reconstructed_coords']

    embeddings = model_output['latent_vec']
    template_embedding = model_output['template_code']
    zero_latent_code = model_output['zero_latent_code']

    sdf_stage1 = model_output['sdf_stage1']
    target2target_output_sdf = model_output['target2target_output_sdf']

    grad_sdf_stage1 = model_output['grad_sdf_stage1']

    # Loss 1
    sdf_constraint_stage1 = torch.where(gt_sdf != -1,
                                        torch.clamp(sdf_stage1, -0.5, 0.5) - torch.clamp(gt_sdf, -0.5, 0.5),
                                        torch.zeros_like(sdf_stage1))
    # Loss 2
    sdf_constraint_stage2 = torch.where(gt_sdf != -1,
                                        torch.clamp(target2target_output_sdf, -0.5, 0.5) - torch.clamp(gt_sdf, -0.5,
                                                                                                       0.5),
                                        torch.zeros_like(target2target_output_sdf))

    # Loss 3
    cos_sim, _, _, grad_norm, _ = DCI_utils.Safe_Cosine_Similarity.apply(grad_sdf_stage1, gt_normals, -1, True, 1e-8,
                                                                         1e-8)
    normal_constraint = torch.where(gt_sdf == 0, 1 - cos_sim, torch.zeros_like(grad_sdf_stage1[..., :1]))

    # Loss 4
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(sdf_stage1), torch.exp(-1e2 * torch.abs(sdf_stage1)))

    # Loss 5
    grad_constraint = torch.abs(grad_norm.squeeze(-1) - 1)

    # Loss 6
    embeddings_constraint = embeddings ** 2
    # Loss 7
    template_reg = (zero_latent_code - template_embedding) ** 2

    # Loss 9 Similarity loss between the template space and instance space?
    similarity_constraint = (embeddings - template_embedding.expand(embeddings.shape[0], -1)) ** 2

    # Loss 8
    reconstructed_loss = reconstructed_coords - coords[..., :3]
    reconstructed_loss = reconstructed_loss ** 2

    return {
        'sdf_constraint_stage1': torch.abs(sdf_constraint_stage1).mean() * 3e3,
        'sdf_constraint_stage2': torch.abs(sdf_constraint_stage2).mean() * 3e3,
        'reconstructed_loss': reconstructed_loss.mean() * 5e4,
        'inter': inter_constraint.mean() * 5e2,
        'normal_constraint': normal_constraint.mean() * 1e2,
        'grad_constraint': grad_constraint.mean() * 5e1,
        'embeddings_constraint': embeddings_constraint.mean() * 1e6,
        'template_reg': template_reg.mean() * 1e6,
        'similarity_constraint': similarity_constraint.mean() * 1e2
    }
