import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession('./onnx_model/flag_embedding_model.onnx')

# 获取模型的输出节点信息
outputs = session.get_outputs()

# 提取输出名称
output_names = [output.name for output in outputs]

print("模型的输出名称：", output_names)