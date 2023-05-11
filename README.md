# Tomography segmentation

Project is under active development

Aim of the project is to compare different models to perform CT liver segmentation. Sample output is visible here:
![image](https://user-images.githubusercontent.com/53475125/223480183-f7baf719-5ad3-43cf-8a37-2abeca30c540.png)

All code is written in PyTorch, wandb for tracking and HuggingFace for transformers. It features few modern solutions:
* Quanet(https://github.com/15029257158/QAU-Net)
* Defednet(https://github.com/SUST-reynole/DefED-Net)
* Vit segmetner(https://github.com/rstrudel/segmenter/tree/master)
* Unet++(https://github.com/MrGiovanni/UNetPlusPlus)
* Polar images(https://github.com/marinbenc/medical-polar-training)

For created experiments ViT segmenter performs best in tumor segmentation and Unet++ for liver segmentation. 
