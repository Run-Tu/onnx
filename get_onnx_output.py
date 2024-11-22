import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


session = ort.InferenceSession("./onnx_model/flag_embedding_model.onnx")
tokenizer = AutoTokenizer.from_pretrained("./models/models--TidalTail--FinQA-FlagEmbedding/snapshots/272edad9ab6cd0160d2baf3597f881c8d49f1de1")


def get_session_outputsName(session):
    # 获取模型的输出节点信息
    outputs = session.get_outputs()
    # 提取输出名称
    output_names = [output.name for output in outputs]
    print("模型的输出名称：", output_names)


def get_session_outputShape(session, tokenizer):
    text = "这是一个测试输入,用于验证模型的推理速度"
    inputs = tokenizer(text, return_tensors='np')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # 推理
    outputs = session.run(
        None, 
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )
    # 获取实际输出形状
    actual_output = outputs[0]
    print(f"输出内容: {len(actual_output)}")
    print(f"实际输出形状: {actual_output.shape}")


if __name__ == '__main__':
    get_session_outputsName(session)
    get_session_outputShape(session, tokenizer)