# ResTT: A Quantum-inspired Model for Learning Multilinear Correlations

This repository contains PyTorch implementation for ResTT.

For more details, please refer to [arXiv](https://arxiv.org/abs/2108.08659) and [T-AI](https://ieeexplore.ieee.org/document/9842296).

**Abstract:**

States of quantum many-body systems are defined in a high-dimensional Hilbert space, where rich and complex interactions among subsystems can be modelled. In machine learning, complex multiple multilinear correlations may also exist within input features. In this work, we present a quantum-inspired multilinear model, named Residual Tensor Train (ResTT), to capture the multiple multilinear correlations of features, from low to high orders, within a single model. ResTT is able to build a robust decision boundary in a high-dimensional space for solving fitting and classification tasks. In particular, we prove that the fully-connected layer and the Volterra series can be taken as special cases of ResTT. Furthermore, we derive the rule for weight initialization that stabilizes the training of ResTT based on a mean-field analysis.

## Usage

### Requirements

- torch>=1.0.0
- torchvision>=0.2.0
- opencv-python==4.5.3


### Training

To train ResTT, run:
```
python main.py
```

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing:
```
@article{chen2022residual,
  title={Residual Tensor Train: A Quantum-inspired Approach for Learning Multiple Multilinear Correlations},
  author={Chen, Yiwei and Pan, Yu and Dong, Daoyi},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2022},
  publisher={IEEE}
}
```
