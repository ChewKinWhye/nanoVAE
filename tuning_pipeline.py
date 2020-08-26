import os

# VAE DNA
os.system("CUDA_VISIBLE_DEVICES=-1 python VAE_DNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 100000 --output_filename VAE_DNA")

# VAE RNA
#os.system("CUDA_VISIBLE_DEVICES=-1 python VAE_RNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 900000 --output_filename VAE_RNA")

print("COMPLETED PIPELINE")

