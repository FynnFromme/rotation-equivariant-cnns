# Rotation Equivariance in Convolutional Neural Networks

Fynn Fromme, Paderborn University

---

This repository contains the code accompanying the Bachelor's thesis titled **"Rotation Equivariance in Convolutional Neural Networks"**. The thesis compares various rotation equivariant architectures:

- **$\mathbf{C_4}$-CNNs** (based on [G-CNNs](https://github.com/tscohen/GrouPy) [[1]](#cohen16))
- **$\mathbf{C_n}$-CNNs** (based on [SE(2,N)-CNNs](https://github.com/tueimage/SE2CNN/tree/master) [[2]](#lafarge21))
- **$\mathrm{\mathbf{SO}}\mathbf{(2)}$-CNNs** (based on [H-Nets](https://github.com/danielewworrall/harmonicConvolutions) [[4]](#worrall17))

In this repository, we will refer to the approaches by their original name. The implementations are based on the original implementations. Their licenses can be found in the corresponding packages.

## Contents

- `data/`: Scripts for loading and visualizing data as well as a notebook, where the MNIST-rot [[3]](#larochelle07) dataset can be explored.
- `networks/`: Contains python packages implementing the equivariant CNNs and the corresponding test modules.
- `experiments/`: Provides the trained models that were evaluated and a Jupyter notebook for each architecture. They show how to define and train the models, and visualize kernels and feature maps interactively.

## Installation

To set up the environment, you will need a working TensorFlow 2.16.1+ installation with Python 3.11.7+. The additional packages can be installed as follows:

```bash
pip install -r requirements.txt
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## References
<a id="cohen16">[1]</a> 
Taco Cohen and Max Welling. “Group Equivariant Convolutional Networks”. In: Proceedings of The 33rd International Conference on Machine Learning. International Conference on Machine Learning. PMLR, June 11, 2016, pp. 2990–2999. url: https://proceedings.mlr.press/v48/cohenc16.html (visited on 02/09/2024).

<a id="lafarge21">[2]</a> 
Maxime W. Lafarge et al. “Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis”. In: Medical Image Analysis 68 (Feb. 1, 2021). issn: 1361-8415. doi: 10.1016/j.media.2020.101849. url: https://www.sciencedirect.com/science/article/pii/S1361841520302139 (visited on 02/09/2024)

<a id="larochelle07">[3]</a> 
Hugo Larochelle et al. “An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation”. In: Proceedings of the 24th International Conference on Machine Learning. ICML ’07 & ILP ’07: The 24th Annual International Conference on Machine Learning Held in Conjunction with the 2007 International Conference on Inductive Logic Programming.Corvalis Oregon USA: ACM, June 20, 2007, pp. 473–480. isbn: 978-1-59593-793-3. doi: 10.1145/1273496.1273556. url: https://dl.acm.org/doi/10.1145/1273496.1273556 (visited on 06/16/2024).

<a id="worrall17">[4]</a> 
Daniel E. Worrall et al. “Harmonic Networks: Deep Translation and Rotation Equivariance”. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017, pp. 5028–5037. url: https://openaccess.thecvf.com/content_cvpr_2017/html/Worrall_Harmonic_Networks_Deep_CVPR_2017_paper.html (visited on 02/09/2024).
