# GasSeg: A Lightweight Real-Time Infrared Gas Segmentation Network for Edge Devices
This is the official code of GasSeg: A Lightweight Real-Time Infrared Gas Segmentation Network for Edge Devices.
## IGS Dataset
We propose a high-quality real-world IGS dataset, containing 6426 images and 7390 segmentation targets.
### Dataset Access  
Due to the dataset being collected from real industrial scenarios, the **training dataset is available upon request**. Please send an email to **22225179@zju.edu.cn**, clearly stating your purpose of use. 
Thank you for your understanding and support of our work!  
To reproduce our code, we provide the IGS dataset for testing. Please download it from [IGSDatasetBaidu](https://pan.baidu.com/s/1-VZkDBa6b1l6q5L27VinEQ?pwd=rxq1)|[IGSDatasetGoogle](https://drive.google.com/file/d/1-hB5uEAOdDjCOMShOEwF2D2oTMYxKFcU/view?usp=drive_link), and place it in the **gasmmsegdata** folder like:
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
