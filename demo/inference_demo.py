import sys

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]


model = init_detector(config_file,checkpoint_file)

for i in range(1,11):
    test_img = './test_'+str(i)+'.jpg'
    result = inference_detector(model,test_img)
    show_result(test_img,model.CLASSES, out_file=test_img+'result')