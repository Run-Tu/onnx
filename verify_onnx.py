import onnxruntime as ort
import numpy as np
import time
import torch
from transformers import AutoTokenizer

model_path = "./onnx_model/flag_embedding_model.onnx"
tokenzier = AutoTokenizer.from_pretrained("./models/models--TidalTail--FinQA-FlagEmbedding/snapshots/272edad9ab6cd0160d2baf3597f881c8d49f1de1")
text = "这是一个测试输入,用于验证模型的推理速度"
inputs = tokenzier(text, return_tensors='np')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']

input_dict = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
}
# CPU session
cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
# GPU session
gpu_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# 预热(Option)
warmup_runs = 5
for _ in range(warmup_runs):
    cpu_session.run(['output'], input_dict)
    gpu_session.run(['output'], input_dict)

# 定义测试函数
def measure_inference_time(session, input_dict, num_runs=100):
    times=[]
    for _ in range(num_runs):
        start_time = time.time()
        outputs = session.run(['output'], input_dict)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / num_runs

    return avg_time, outputs

# 测试CPU推理时间
cpu_avg_time, cpu_outputs = measure_inference_time(cpu_session, input_dict)
print(f"CPU平均推理时间: {cpu_avg_time * 1000:.2f} ms")
print(f"CPU_OUTPUTS: {cpu_outputs}")

# 测试GPU推理时间
gpu_avg_time, gpu_outputs = measure_inference_time(gpu_session, input_dict)
print(f"GPU平均推理时间: {gpu_avg_time * 1000:.2f} ms")
print(f"GPU_OUTPUTS: {gpu_outputs}")

# **计算加速比**
if gpu_avg_time > 0:
    speedup = cpu_avg_time / gpu_avg_time
    print(f"GPU 相对于 CPU 的加速比：{speedup:.2f}x")
else:
    print("GPU 推理时间为 0，可能存在问题。")

# **验证模型输出（可选）**
# 比较 CPU 和 GPU 的输出是否一致
if np.allclose(cpu_outputs[0], gpu_outputs[0], atol=1e-5):
    print("CPU 和 GPU 的模型输出一致。")
else:
    print("CPU 和 GPU 的模型输出不一致。")