import os
import torch

# Ask the user for the MIG GPU Instance UUID
gpu_instance_uuid = input("Enter the MIG GPU Instance UUID: ")

# Set the environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_instance_uuid

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of CUDA devices
    num_devices = torch.cuda.device_count()
    print(f'Number of CUDA devices: {num_devices}')

    # List all CUDA devices
    for i in range(num_devices):
        device = torch.cuda.get_device_name(i)
        print(f'Device {i}: {device}')
else:
    print('CUDA is not available.')
