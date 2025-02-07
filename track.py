
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':
    # model = RTDETR('weights/rtdetr-r18.pt') # select your model.pt path
    model = RTDETR('runs/train/exp10/weights/best.pt') # select your model.pt path
    model.track(source='222.mp4',
                project='runs/track',
                name='exp11',
                save=True,
                tracker="botsort.yaml"
                )