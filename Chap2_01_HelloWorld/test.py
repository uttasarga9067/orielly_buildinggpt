# from openai import OpenAI

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="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",
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )

# import torch

# print("Torch version:",torch.__version__)

# print("Is CUDA enabled?",torch.cuda.is_available())

import torch
# print(f'PyTorch version: {torch.__version__}')
# print('*'*10)
# print(f'_CUDA version: ')
# !nvcc --version
# print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')


print(torch.cuda.memory_summary(device={torch.cuda.get_device_name()}, abbreviated=False))


