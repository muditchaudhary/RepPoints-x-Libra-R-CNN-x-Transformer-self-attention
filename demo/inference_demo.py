import sys

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]
out_file = sys.argv[3]
test_img = './test.jpg'
model = init_detector(config_file,checkpoint_file)

result = inference_detector(model,test_img)
show_result(test_img,model.CLASSES, out_file=out_file)