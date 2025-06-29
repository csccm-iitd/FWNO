# FWNO
Generative flow induced neural architecture search: Towards discovering optimal architecture in wavelet neural operator

+ This repository also covers the codes of the following paper 
  > + Thakur, A., Tripura, T., & Chakraborty, S. (2022). Multi-fidelity wavelet neural operator with application to uncertainty quantification. arXiv preprint arXiv:2208.05606. [Article](https://arxiv.org/pdf/2208.05606)

## Files and Folders
```
📂 Burgers                     # 1D Burgers equation experiments and utilities
  |_ GFlowNet_Initializations.py      # FWNO with different initializations
  |_ GFlowNet_LargeDataset.py         # FWNO on larger datasets
  |_ GFlowNet.py                      # FWNO for Burgers
  |_ MCTS.py                         # MCTS based NAS for Burgers
  |_ Random_Search.py                # Random search baseline for architecture search
  |_ Super_Resolution_Plot.py        # Super-resolution plotting
  |_ super_utils.py                  # Utility functions for super-resolution (for future work)
  |_ super_wavelet_convolution.py    # Wavelet convolution operations (for future work)
  |_ utilities3.py                   # Utility fuctions like  dataloader and loss functions
  |_ wno_1d_Burgers_super.py         # Super-resolution with Wavelet Neural Operator (for future work)
  |_ wno_1d_Burgers.py               # WNO for 1D Burgers
  |_ 📂 Data                         # Folder to store data for the Burgers Equation

📂 Darcy                       # 2D Darcy flow experiments and utilities
  |_ GFlowNet.py                      # FWNO for Darcy
  |_ MCTS.py                         # MCTS based NAS for Darcy
  |_ utilities3.py                   # Utility fuctions like  dataloader and loss functions
  |_ wno_2d_darcy.py                 # WNO for 2D Darcy flow
  |_ 📂 Data                         # Folder to store data for the Darcy Equation  in Rectangular Domain

📂 Darcy_notch                 # 2D Darcy flow in triangular domain with a notch
  |_ GFlowNet.py                      # FWNO for Darcy notch
  |_ MCTS.py                         # MCTS based NAS for Darcy notch
  |_ utilities3.py                   # Utility fuctions like  dataloader and loss functions
  |_ wno_2d_Darcy_notch.py           # WNO for 2D Darcy notch
  |_ 📂 Data                         # Folder to store data for the Darcy Equation in Triangular Domain and with a notch

📂 Navier_Stokes               # 2D time-dependent Navier-Stokes experiments and utilities
  |_ GFlowNet.py                      # FWNO for Navier-Stokes
  |_ MCTS.py                         # MCTS based NAS for Navier-Stokes
  |_ utilities3.py                   # Utility fuctions like  dataloader and loss functions
  |_ wno_2d_time_NS.py               # WNO for 2D time-dependent Navier-Stokes
  |_ 📂 Data                         # Folder to store data for the Navier-Stokes Equation
```

## Essential Python Libraries
The following packages are required to be installed to run the above codes:
  + numpy: https://numpy.org/
  + scipy: https://scipy.org/
  + matplotlib: https://matplotlib.org/
  + h5py: https://www.h5py.org/
  + PyTorch: https://pytorch.org/
  + PyWavelets: https://pywavelets.readthedocs.io/en/latest/
  + pytorch_wavelets: https://github.com/fbcotter/pytorch_wavelets
  + monte-carlo-tree-search: https://pypi.org/project/monte-carlo-tree-search/2.0.5/
