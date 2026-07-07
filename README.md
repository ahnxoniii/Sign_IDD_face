# Sign_IDD_face  

This repository is the official PyTorch implementation of **"Facial Keypoint-Based Sign Language Generation Method for Incorporating Non-Manual Components"**  
This paper published 대한전자공학회 하계학술대회, 2026.06



----
Soeun An, Jinsun Park(Major Professor)  
**VIP(Visual Intelligence and Perception) Lab / Pusan Nat'l Univ Computer Science and Engineering / Busan, Republic of Korea**  

## Abstract
Sign language is a visual language that conveys meaning through both manual components, such as hand shape and movement, and non-manual components, such as facial expressions and gaze. Existing sign language generation methods mainly focus on generating manual components from linguistic inputs, while non-manual components are not sufficiently considered. To overcome this limitation, we propose an extended sign language generation method based on the diffusion-based model Sign-IDD. Our method incorporates facial keypoints into the pose representation and applies a separate face joint loss to reflect the different characteristics of body, hand, and face regions. Experimental results show that our method improves the test BLEU-1 score from 24.08 to 25.27 and the ROUGE score from 23.61 to 25.34 compared with the baseline model. These results indicate that incorporating facial non-manual components through facial keypoints improves the semantic delivery and expressiveness of generated poses. 

# Training
```text
python __main__.py train ./Configs/Sign-IDD.yaml
```

# Inference
```text
python __main__.py test ./Configs/Sign-IDD.yaml
```

# SLT Model
We use the back translation [SLT](https://github.com/NaVi-start/Sign-IDD-SLT.git).
# Reference
If you use this code in your research, please cite the following [papers](https://arxiv.org/abs/2412.13609):

```bibtex
@inproceedings{tang2025sign,
  title={Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production},
  author={Tang, Shengeng and He, Jiayi and Guo, Dan and Wei, Yanyan and Li, Feng and Hong, Richang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7266--7274},
  year={2025}
}

@article{tang2024discrete,
  title={Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observation},
  author={Tang, Shengeng and He, Jiayi and Cheng, Lechao and Wu, Jingjing and Guo, Dan and Hong, Richang},
  journal={arXiv preprint arXiv:2411.16810},
  year={2024}
}

@article{tang2024GCDM,
  title={Gloss-Driven Conditional Diffusion Models for Sign Language Production},
  author={Tang, Shengeng and Xue, Feng and Wu, Jingjing and Wang, Shuo and Hong, Richang},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications},
  issn = {1551-6857},
  year={2024},
}
```


# Acknowledge
This work was supported by the National Natural Science Foundation of China (Grants No. U23B2031, 61932009, U20A20183, 62272144, 62302141, 62331003), the Anhui Provincial Natural Science Foundation, China (Grant No. 2408085QF191), the Major Project of Anhui Province (Grant No. 202423k09020001), and the Fundamental Research Funds for the Central Universities (Grants No. JZ2024HGTA0178, JZ2024HGTB0255).
