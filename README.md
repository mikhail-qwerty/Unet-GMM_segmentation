# Machine learning approach to quantitative analysis for dynamic CT images of methane hydrate formation in sandy samples
by
Mikhail I. Fokin, Viktor V. Nikitin and Anton A. Duchkov

## Content

- data.py:
(file contains data processing functions)
	
- IO.py:
(file contains functions for saving and loading GMM models)

- segmentation.py:
(file contains functions for segmentation using U-net and GMM models)

- Unet_models.py:
(file contains definition of 2D and 3D U-net models)

- train_models.py:
(example of U-nets training)
	
- dataset_segmentationipynb:
(example of using two step segmentation algorithm)
	
- dynamic_segmentation.ipynb:
(example of using two step segmentation algorithm for dynamic data). 
	
  ## Dependencies

- Python (version 3.8)

- tensorflow (2.7.0)
- numpy (1.17.1)
- scipy (1.1.0)
- scikit-learn (0.22.1)
- matplotlib (3.1.1)
- tqdm (4.63.0)
- opencv-python (4.5.5.64)
- h5py (3.6.0)
- tifffile (2022.3.16)
  ## Licens

BSD 3-Clause License

Copyright (c) 2022, Mikhail I. Fokin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
