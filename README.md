# GasSeg: A Lightweight Edge-Guided and Context-Aware Network for Real-Time Infrared Gas Segmentation
This is the official code of GasSeg: A Lightweight Edge-Guided and Context-Aware Network for Real-Time Infrared Gas Segmentation.
## IGS Dataset
We propose a high-quality real-world IGS dataset, containing 6426 images and 7390 segmentation targets.
### Dataset Access  
Due to the unique characteristics of the infrared gas dataset, the **training dataset is available upon request**. Please send an email to **22225179@zju.edu.cn**, clearly stating your purpose of use. 
Thank you for your understanding and support of our work!  
Prepare our IGS dataset: Please download it from [IGSDataset](), and place it in the gasmmsegdata folder.
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
| Model       | Dataset     | mIoU (%) | mF1 (%) | Download Link |
|-------------|-------------|----------|---------|---------------|
| GasSeg-M    | IGS Dataset | 90.68    | 94.09   | [Download](#) |
| GasSeg-S    | IGS Dataset | 89.06    | 95.02   | [Download](#) |
| GasSeg-L    | IGS Dataset | 90.88    | 95.14   | [Download](#) |
### Setup
For detailed setup instructions, we recommend referring to the [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation).
```
conda create -n GasSeg python==3.10
conda activate GasSeg
pip install -r requirements.txt
```
### Train
```
python tools/train.py configs/gasseg/gasseg-m.py
```
### Test
```
python tools/test.py configs/gasseg/gasseg-m.py models/GasSeg-m.pth
```
## GasSeg
![ Illustration the architecture of GasSeg.]

## Contact   
For any question, feel free to email <22225179@zju.edu.cn>

### Acknowledgments
We would like to thank the developers of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for their open-source contributions, which greatly supported the development of our work.
