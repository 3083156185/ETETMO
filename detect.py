import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/ETETVA/weights/best_plane.pt') # select your model.pt path
    model.predict(source='dataset/images/test/2.png',
                  conf=0.25,
                  project='runs/detect',
                  name='exp30',
                  save=True,
                  visualize=False# visualize model features maps
                  )