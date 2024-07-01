# Python Program to Create Dataset for PaddleOCR's Text Detection Algorithm DBNet and YOLO

## For Deployment
```bash
python main.py --type=paddle --dir=C:/Users/vinci/OneDrive/Desktop/Learn/Github/yolo/ --n=5x --rotate=-30to30 --zrot=yes --blur=1to6 --darken=40to50 --dis=5 --brighten=40to50 --elastic=500to600 --rigid=10to15 --lbcor=416,416 --percent=70
```
## NOTE:

x is times

rotate should be a multiple of 5 

blur should be a multiple of 3

darken/brighten should be a multiple of 10

For OCR:
type=paddle

For YOLO:
type=yolo


