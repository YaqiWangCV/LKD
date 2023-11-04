# LKD

This repo is the official implementation of 'Lesion-aware Knowledge Distillation for Diabetic Retinopathy Lesion Segmentation'.

**The manuscript is now under review.**

![framework](https://github.com/YaqiWangCV/LKD/main/docs/framework.jpg)

## Requirements

Install from the ```requirements.txt``` using:

```angular2html
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

IDRiD and DDR datasets can be downloaded in the following links:

* IDRiD Dataset - [Link](https://idrid.grand-challenge.org/)
* DDR Dataset - [Link](https://github.com/nkicsl/DDR-dataset)

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── data
│   ├── idrid
│   ├── ddr
```

The DDR dataset is only required to the segmentation part without including the grading part.

### 2. Training

The first step is to train the teacher network. You can select your desired type of teacher network in ```train_single_network.py```.

Run:

```angular2html
python train_single_network.py
```

Also different segmentation networks can be trained by this script, they are 'MCA-UNet', 'U²Net', 'UNet', 'UNet++', 'UCtransnet', 'UDtransnet', 'Attention-UNet', 'Res-UNet++', 'UNet3+', 'TransUNet' and 'ENet'. 



Once the teacher network is well-trained, it can be used to guide the student network. Just set ```teacher_checkpoints``` to the path of your teacher network weights in ```train_student_lkd.py``` and then Run:

```angular2html
python train_student_lkd.py
```
### 3. Testing

 Run:

```angular2html
python test.py
```

You can get the Dice、IoU and AUPR scores. 



<!--
## Citations

If this code is helpful for your study, please cite:
```
@misc{wang2021uctransnet,
      title={UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer}, 
      author={Haonan Wang and Peng Cao and Jiaqi Wang and Osmar R. Zaiane},
      year={2021},
      eprint={2109.04335},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
-->

## Contact

Yaqi Wang ([wangyaqicv@gmail.com](wangyaqicv@gmail.com))
