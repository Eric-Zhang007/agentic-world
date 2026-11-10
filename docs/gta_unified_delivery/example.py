from pathlib import Path
from dataset import load_dataset

root = Path(__file__).resolve().parent
dataset = load_dataset(root, manifest='train', temporal_indices=[0,30,60,90,119])
sample = dataset[0]
print('records:', len(dataset))
print('sample:', sample['sample_id'], sample['task'])
print('whitebox:', sample['conditions']['target_whitebox'].shape)
print('RGB supervision:', sample['supervision']['target_rgb'].shape)
print('reference pairs:', len(sample['conditions']['reference_pairs']))
