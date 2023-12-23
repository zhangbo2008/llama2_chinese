import torch

# 初始化Half Tensor
h = torch.tensor([1.0,2.0,3.0], dtype=torch.half)
# h = torch.tensor([1.0,2.0,3.0], dtype=torch.float16) # 跟上面一行一样.

# 查看数据类型
print(h.dtype)
import accelerate
import bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
from transformers import AlbertTokenizer, AlbertModel
model = AlbertModel.from_pretrained('./albert',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True,low_cpu_mem_usage=True)
# torch_dtype 模型本身的类型, 不写的话就自己根据权重文件查询出来.这个是权重文件本身决定的,一般在config.json里面
# load_in_8bit 会把模型转化为8bit类型.这个可以自己设置.

print(1)