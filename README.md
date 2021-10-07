# Low-Light-Enhancement

Low light image enhancement using [MIRNet](https://arxiv.org/abs/2003.06792) in PyTorch. <br> <br>
Check out my [blog](https://vrushank98.hashnode.dev/low-light-image-enhancement-using-neural-networks) for thorough understanding.

## Reults:

`This model achieves 22.97 PSNR on LoL dataset's evaluation set.`
<br>
Input image:point_right: Restored/Predicted Image :point_right: Ground Truth image 

![img1](https://github.com/Vrushank264/Low-Light-Enhancement/blob/main/Results/result1.png)
![img2](https://github.com/Vrushank264/Low-Light-Enhancement/blob/main/Results/result6.png)
![img3](https://github.com/Vrushank264/Low-Light-Enhancement/blob/main/Results/result4.png)
![img4](https://github.com/Vrushank264/Low-Light-Enhancement/blob/main/Results/result7.png)

## Details:

1. This model uses [LoL dataset](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view).
2. All the images were resized to 64x64 during training.
3. It uses 3 Residual Recurrent Groups(RRGs) where every RRG contains 2 Multi-scale residual blocks(MSRBs).
4. You can see more results in `Results` directory.


