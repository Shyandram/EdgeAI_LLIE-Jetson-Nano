import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt
import time

#sess = rt.InferenceSession("weights/OV_enhance_color-llie-ResCBAM_g-AEFT-256.onnx", providers=rt.get_available_providers())
sess = rt.InferenceSession("weights/OV_enhance_color-llie-ResCBAM_g-AEFT-256.onnx", providers=["CUDAExecutionProvider"])

input_name = sess.get_inputs()[0].name
print("input name", input_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_name = sess.get_outputs()[0].name
print("output name", output_name)

BATCH_SIZE = 1

# img  = cv2.imread("../cloudy (1)_bmp.bmp")
# img = resize(io.imread(url), (224, 224))
# url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
# img = resize(io.imread(url), (256, 256))
url = "../Test/LIME/5.bmp"
img = cv2.resize(np.asarray(Image.open(url).convert('RGB')), (256, 256))
img = np.transpose(img, (2, 0, 1)) /255.
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension


pred = sess.run([output_name], {input_name: img})[0][0]
print(pred.shape)
pred = np.transpose(pred, [1, 2, 0])*255.
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR) 
cv2.imwrite("out_onnx.jpg", pred)


total_time = 0
start_t = time.time()
itr_cnt = int(1e2)
for step in enumerate(range(itr_cnt)):
	sess.run([output_name], {input_name: img})[0][0]
end_t = time.time()
print("Average time (total) taken by network is : %f ms"%((end_t-start_t)*1e3/itr_cnt))
