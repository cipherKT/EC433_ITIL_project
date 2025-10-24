# GATv2 Extension: Reproduction and Enhancement Study

This repository contains the implementation code for our reproduction and extension study of Graph Attention Networks v2 (GATv2), based on the paper ["How Attentive are Graph Attention Networks?"](https://arxiv.org/abs/2105.14491) by Brody et al.

##  Paper Overview

Graph Neural Networks (GNNs) have emerged as powerful tools for learning on graph-structured data. While the original Graph Attention Network (GAT) introduced attention mechanisms to graphs, it suffered from a static attention limitation. **GATv2** addresses this by introducing dynamic attention mechanisms that adapt more effectively to different contexts.

Our work reproduces the key findings from the original GATv2 paper and extends the study by:
- Testing on additional real-world datasets (Amazon co-purchase network)
- Implementing architectural variations with residual connections and multi-head dynamic attention
- Conducting comprehensive hyperparameter sensitivity analysis
- Visualizing learned attention weights for interpretability

##  Key Contributions

1. **Reproduction Study**: Verified GATv2's improvements over GAT on standard citation datasets
2. **Extended Evaluation**: Applied GATv2 to large-scale real-world graph (Amazon0302)
3. **Architectural Enhancements**: 
   - Residual connections for improved training stability
   - Multi-head dynamic attention mechanisms
4. **Hyperparameter Analysis**: Systematic study of learning rates, hidden dimensions, and attention heads
5. **Visualization**: Attention weight visualization for model interpretability

##  Results

Our experiments demonstrate that:
- **Residual connections** significantly improve training stability and convergence
- **Optimal hyperparameters**: Learning rate of 0.005 with hidden size of 32 yields best performance
- **Attention visualization** provides valuable insights into the model's decision-making process

![Validation Accuracy Heatmap](result.png)
*Validation accuracy across different hyperparameter configurations*

## Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx matplotlib seaborn
```

### Running the Code

1. Clone the repository:
```bash
git clone https://github.com/cipherKT/EC433_ITIL_project.git
cd EC433_ITIL_project
```

2. Open the Jupyter notebook:
```bash
jupyter notebook GAT2.ipynb
```

3. Run all cells to reproduce the experiments

### Dataset

The code uses the Amazon0302 co-purchase network dataset. The dataset will be automatically loaded when you run the notebook.

##  Implementation Details

- **Model Architecture**: Two-layer GATv2 with residual connections
- **Dataset**: Amazon0302 (262,111 nodes, 899,792 edges)
- **Features**: 64-dimensional random features
- **Classes**: 10 synthetic classes
- **Split**: 60% training, 20% validation, 20% test

### Hyperparameters Explored

| Parameter       | Values Tested      |
| --------------- | ------------------ |
| Learning Rate   | 0.001, 0.005, 0.01 |
| Hidden Size     | 8, 16, 32          |
| Attention Heads | 2, 4, 8            |
| Dropout         | 0.6                |
| Weight Decay    | 5×10⁻⁴             |

##  References
```bibtex
@article{brody2021attentive,
  title={How Attentive are Graph Attention Networks?},
  author={Brody, Shaked and Alon, Uri and Yahav, Eran},
  journal={arXiv preprint arXiv:2105.14491},
  year={2021}
}
```

##  Authors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pragati-paraagi">
        <img src="https://github.com/pragati-paraagi.png" width="100px;" alt="Pragati Agrahari"/><br />
        <sub><b>Pragati Agrahari</b></sub>
      </a><br />
      <sub>202251097</sub>
    </td>
    <td align="center">
      <a href="https://github.com/suryansh-sahay">
        <img src="https://github.com/suryansh-sahay.png" width="100px;" alt="Suryansh Sahay"/><br />
        <sub><b>Suryansh Sahay</b></sub>
      </a><br />
      <sub>202251137</sub>
    </td>
    <td align="center">
      <a href="https://github.com/cipherKT">
        <img src="https://github.com/cipherKT.png" width="100px;" alt="Kunj Thakkar"/><br />
        <sub><b>Kunj Thakkar</b></sub>
      </a><br />
      <sub>202251142</sub>
    </td>
  </tr>
</table>


##  Full Report

For detailed analysis, methodology, and results, please refer to our [full technical report](https://drive.google.com/file/d/1h0Nulb8e4jCCsPC_2Vrao7E5GVg-Wskb/view?usp=sharing).

##  Related Links

- [Original GATv2 Paper](https://arxiv.org/abs/2105.14491)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Amazon0302 Dataset](http://snap.stanford.edu/data/amazon0302.html)

## Acknowledgments

- Original GATv2 authors for their groundbreaking work
- IIIT Vadodara for supporting this research
- PyTorch Geometric team for their excellent library

---

*This project was completed as part of the EC433 ITIL course at IIIT Vadodara.*
