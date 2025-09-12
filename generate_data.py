import random
from datasets import load_dataset
data_dir = "/home/nicoleg/physionet.org/files/mimic-cxr-jpg/2.0.0/"

dataset = load_dataset('vllm-pneumonia-detection/mimic-cxr-labelled-dataset')
print("Loaded dataset:")
print(dataset)

simple_labels = ["this is a photo of no pneumonia", "this is a photo of pneumonia"]

# possible labels to try
medical_labels = [
"""
FINDINGS:

The lung fields are clear and without focal airspace consolidation, mass, or nodule. There is no evidence of pleural effusion or pneumothorax.
The cardiomediastinal silhouette is within normal limits for size and contour.
The visualized osseous structures, including the thoracic cage, are intact and without acute fracture or destructive lesion. The soft tissues are unremarkable.
The trachea is midline.


IMPRESSION:

No acute cardiopulmonary abnormality.
Normal chest radiograph.
""",

"""
FINDINGS:

There is a dense, homogenous opacity consistent with consolidation.
Within this region, air bronchograms are noted, which appear as darker, branching, 
air-filled structures against the white background of the fluid-filled lung tissue.

IMPRESSION:

Pneumonia. Clinical correlation is recommended. Follow-up imaging 
may be suggested to assess resolution after treatment.
"""
]


labels = simple_labels
print(labels)

def process(x):
    image = data_dir + x['image_files']
    label = labels[0] if x['chexagent'] == 0 else labels[1]
    return {'image': image, 'label': label}

dataset = dataset.map(process)
print("Processed dataset:")
print(dataset)
print(random.choice(dataset['train']))

dataset['train'].to_csv('train_data.csv')
dataset['validate'].to_csv('validate_data.csv')
