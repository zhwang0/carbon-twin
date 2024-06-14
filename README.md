# Carbon-Twin Dataset

## Overview
The Carbon-Twin dataset is a comprehensive collection of remote sensing data focused on carbon cycle processes. This dataset is designed to support research in climate science, environmental monitoring, and ecological studies. By providing detailed observations and measurements, the Carbon-Twin dataset aims to enhance our understanding of carbon dynamics and their impact on global climate change.

## Features
- **Global-Scale Coverage**: The first global-scale machine learning-ready dataset for monitoring and forecasting forest carbon dynamics, provided at a 0.5° resolution.
- **Extended Temporal Range**: Spans over 40 years, offering a comprehensive historical perspective for long-term analysis.
- **Extensive Variable Set**: Covers over 100+ variables integrated from heterogeneous sources, ensuring a rich and diverse dataset.
- **Scenario-Based Training and Testing**: Includes different scenarios to resemble diverse application conditions, enhancing the dataset's versatility for various research needs.
- **Benchmark Experiments**: Provides benchmark experiments with a suite of machine learning forecasting models and new metrics specifically designed for the carbon forecasting problem.


## Benchmarks
The repository includes codes for the following methods:
- **LSTM**: A standard LSTM taking time-series ED inputs and outputting targets [Graves, 2012](https://www.cs.toronto.edu/~graves/phd.pdf).
- **LSTNet**: A time-series model extracting short-term dependency using convolution along the time-feature dimension and long-term temporal patterns using RNN [Lai et al., 2018](https://arxiv.org/abs/1703.07015).
- **DeepED**: An LSTM-based deep learning emulator for ED with specialized designs on error accumulation reduction and knowledge-guided learning [Wang et al., 2023](https://doi.org/10.1145/3589132.3625577).
- **Transformer (TF)**: A vanilla transformer model with a self-attention mechanism to capture dependencies across all time steps during the forecasting [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
- **Informer (IF)**: A transformer variant with a ProbSparse attention for reduced complexity [Zhou et al., 2021](https://arxiv.org/abs/2012.07436).
- **DLinear (DL)**: A linear decomposition model to separate data into trend and seasonality, showing comparable performance among transformer-based models while maintaining efficiency [Zeng et al., 2023](https://arxiv.org/abs/2205.13504).
- **Crossformer (CF)**: A transformer variant with a two-stage attention mechanism for modeling both cross-time and cross-feature dependencies [Zhang et al., 2023](https://arxiv.org/abs/2303.11337).



## Usage
### Accessing the Dataset
The dataset can be publicly available after publication.


## Contact

