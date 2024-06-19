# DGCI

<div align=center><img src="figs/Main.png"></div>

[**_Implicit Representation Embraces ChallengingAttributes of Pulmonary Airway Tree Structures_**]()

> By Minghui Zhang, Hanxiao Zhang, Xin You, Guang-Zhong Yang and Yun Gu
>> Institute of Medical Robotics, Shanghai Jiao Tong University
>> Department of Automation, Shanghai Jiao Tong University, Shanghai, China

## Introduction
> High-fidelity modeling of the pulmonary airway tree from
CT scans is critical to preoperative planning. However, the granularity
of CT scan resolutions and the intricate topologies limit the accuracy of
manual or deep-learning-based delineation of airway structures, resulting
in coarse representation accompanied by spike-like noises and disconnectivity
issues. To address these challenges, we introduce a Deep Geometric
Correspondence Implicit (DGCI) network that implicitly models
airway tree structures in the continuous space rather than discrete
voxel grids. DGCI first explores the intrinsic topological features shared
within different airway cases on top of implicit neural representation
(INR). Specifically, we establish a reversible correspondence flow to constrain
the feature space of training shapes. Moreover, implicit geometric
regularization is utilized to promote a smooth and high-fidelity representation
of fine-scaled airway structures. By transcending voxel-based
representation, DGCI acquires topological insights and integrates geometric
regularization into INR, generating airway tree structures with
state-of-the-art topological fidelity. Detailed evaluation results on the
public dataset demonstrated the superiority of the DGCI in the scalable
delineation of airways and downstream applications.


## Introduction