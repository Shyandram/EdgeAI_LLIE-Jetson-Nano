import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
# from skimage import io
# from skimage.transform import resize
from PIL import Image
import time

# trtexec --onnx=OV_enhance_color-llie-ResCBAM_g-256.onnx --saveEngine=CPGANet_engine.trt --explicitBatch --workspace=128
# trtexec --onnx=OV_enhance_color-llie-ResCBAM_g-256.onnx --saveEngine=CPGANet_engine.trt --explicitBatch --workspace=128 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16

USE_FP16 = False
target_dtype = np.float16 if USE_FP16 else np.float32

f = open("weights/CPGANet_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

BATCH_SIZE = 1

# img  = cv2.imread("../cloudy (1)_bmp.bmp")
# img = resize(io.imread(url), (224, 224))
# url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
# img = resize(io.imread(url), (256, 256))
url = "../Test/LIME/5.bmp"
img = cv2.resize(np.asarray(Image.open(url).convert('RGB')), (256, 256))
img = np.transpose(img, (2, 0, 1)) #/256.
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension

# need to set input and output precisions to FP16 to fully enable it
dummy_name = 'output'
dummy_shape = (BATCH_SIZE, 3, 256, 256)
input_batch = np.empty([BATCH_SIZE, 3, 256, 256], dtype = target_dtype) 
output = np.empty([BATCH_SIZE, 3, 256, 256], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()
def predict(batch): 
	# transfer input data to device
	cuda.memcpy_htod_async(d_input, batch, stream)
	# execute model
	context.execute_async_v2(bindings, stream.handle, None)
	# transfer predictions back
	cuda.memcpy_dtoh_async(output, d_output, stream)
	# syncronize threads
	stream.synchronize()
	return output

pred = predict(input_batch)[0]
pred = np.transpose(pred, [1, 2, 0])*256.
cv2.imwrite("out_trt.jpg", pred)


total_time = 0
start_t = time.time()
itr_cnt = int(1e2)
for step in enumerate(range(itr_cnt)):
	predict(input_batch)
end_t = time.time()
print("Average time (total) taken by network is : %f ms"%((end_t-start_t)*1e3/itr_cnt))
