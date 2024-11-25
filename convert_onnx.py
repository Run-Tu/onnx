import onnx
import torch
from transformers import AutoModel, AutoTokenizer
from model import FlagModel

model_name = "./models/models--TidalTail--FinQA-FlagEmbedding/snapshots/272edad9ab6cd0160d2baf3597f881c8d49f1de1"
model = FlagModel(model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()确保Dropout和BatchNorm在导出时以推理模式运行
device = torch.device('cpu')
model.to(device)
model.eval()

# 构造示例输入
text = "这是一个示例输入"
inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']

onnx_model_path = "./onnx_model/flag_embedding_model.onnx"
torch.onnx.export(
    model=model,
    args=(input_ids, attention_mask, token_type_ids),
    f=onnx_model_path,
    export_params=True,
    do_constant_folding=True,
    input_names=list(inputs.keys()),
    output_names=['output'],
    dynamic_axes ={
        'input_ids':{0:'batch_size', 1:'sequence_length'},
        'attention_mask':{0:'batch_size', 1:'sequence_length'},
        'token_type_ids':{0:'batch_size', 1:'sequence_length'},
        'output':{0:'batch_size',1:'sequence_length'}
    }
)

print(f"模型已成功导出为onnx")