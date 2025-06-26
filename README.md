# GasSeg: A Lightweight Real-Time Infrared Gas Segmentation Network for Edge Devices
This is the official code of GasSeg: A Lightweight Real-Time Infrared Gas Segmentation Network for Edge Devices. Accepted by Pattern Recognition.
## IGS Dataset
We propose a high-quality real-world IGS dataset, containing 6426 images and 7390 segmentation targets.
### Dataset Access  
Due to the dataset being collected from real industrial scenarios, the **training dataset is available upon request**. Please send an email to **22225179@zju.edu.cn**, clearly stating your purpose of use. 
Thank you for your understanding and support of our work!  
To reproduce our code, we provide the test set of IGS dataset for testing. Please download it from [IGSDatasetBaidu](https://pan.baidu.com/s/1-VZkDBa6b1l6q5L27VinEQ?pwd=rxq1)|[IGSDatasetGoogle](https://drive.google.com/file/d/1-hB5uEAOdDjCOMShOEwF2D2oTMYxKFcU/view?usp=drive_link), and place it in the **gasmmsegdata** folder like:
- `gasmmsegdata/`: Root directory for the dataset.
  - `img_dir/`: Contains the input infrared gas images.
    - `train/`: Training set images.
    - `val/`: Validation set images.
  - `ann_dir/`: Contains the corresponding annotated masks.
    - `train/`: Masks for training set.
    - `val/`: Masks for validation set.


## GasSeg Code
### Pre-trained Model Weights 
We provide the pre-trained weights for GasSeg, which can be used directly for testing or fine-tuning.  
#### Available Models  
| Model       | Dataset     | mIoU (%) | mF1 (%) | FPS on 4090 GPU| Download Link |
|-------------|-------------|----------|---------|--------------- |---------------|
| GasSeg-M    | IGS Dataset | 90.68    | 95.02   |215.3| [Baidu](https://pan.baidu.com/s/1TbOuM8yo0ZfAwnLeoCd7Xw?pwd=vmba) \| [Google](https://drive.google.com/file/d/1RZQ5AaWgEV6MMO1SLvHOKhtWLVjFOttg/view?usp=drive_link) |
| GasSeg-S    | IGS Dataset | 89.06    | 94.09   |230.8| [Baidu](https://pan.baidu.com/s/1ooNMbFHmdojCm3-r0WNJmA?pwd=26b3) \| [Google](https://drive.google.com/file/d/1RZQ5AaWgEV6MMO1SLvHOKhtWLVjFOttg/view?usp=drive_link) |
| GasSeg-L    | IGS Dataset | 90.88    | 95.14   |160.7| [Baidu](https://pan.baidu.com/s/17zOqLvPzQ6-_7U6au4U_dw?pwd=itx4) \| [Google](https://drive.google.com/file/d/1Y56JGj2zsr7FSk7LAL_SiLWSyDFoikkj/view?usp=drive_link) |
### Setup
For detailed setup instructions, we recommend referring to the [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation). Our test environment is torch2.3.1+cu11.8 and mmcv2.2.0.
```
conda create -n GasSeg python==3.10
conda activate GasSeg
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.10.5
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
cd GasSeg
pip install -v -e .
```
### Train
```
python tools/train.py configs/gasseg/gasseg-m.py
```
### Test
```
python tools/test.py configs/gasseg/gasseg-m.py models/GasSeg-m.pth
```
## Contact   
For any question, feel free to email <22225179@zju.edu.cn>

### Acknowledgments
We would like to thank the developers of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for their open-source contributions, which greatly supported the development of our work.
Please give us a STAR if the dataset and code help you!
```
@article{YU2026111931,
title = {GasSeg: A lightweight real-time infrared gas segmentation network for edge devices},
journal = {Pattern Recognition},
volume = {170},
pages = {111931},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111931},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325005916},
author = {Huan Yu and Jin Wang and Jingru Yang and Kaixiang Huang and Yang Zhou and Fengtao Deng and Guodong Lu and Shengfeng He},
keywords = {Infrared gas segmentation, Boundary guidance network, Contextual attention, Real-time efficiency, Engineering application},
abstract = {Infrared gas segmentation (IGS) focuses on identifying gas regions within infrared images, playing a crucial role in gas leakage prevention, detection, and response. However, deploying IGS on edge devices introduces strict efficiency requirements, and the intricate shapes and weak visual features of gases pose significant challenges for accurate segmentation. To address these challenges, we propose GasSeg, a dual-branch network that leverages boundary and contextual cues to achieve real-time and precise IGS. Firstly, a Boundary-Aware Stem is introduced to enhance boundary sensitivity in shallow layers by leveraging fixed gradient operators, facilitating efficient feature extraction for gases with diverse shapes. Subsequently, a dual-branch architecture comprising a context branch and a boundary guidance branch is employed, where boundary features refine contextual representations to alleviate errors caused by blurred contours. Finally, a Contextual Attention Pyramid Pooling Module captures key information through context-aware multi-scale feature aggregation, ensuring robust gas recognition under subtle visual conditions. To advance IGS research and applications, we introduce a high-quality real-world IGS dataset comprising 6,426 images. Experimental results demonstrate that GasSeg outperforms state-of-the-art models in both accuracy and efficiency, achieving 90.68% mIoU and 95.02% mF1, with real-time inference speeds of 215 FPS on a GPU platform and 62 FPS on an edge platform. The dataset and code are publicly available at: https://github.com/FisherYuuri/GasSeg.}
}
```
