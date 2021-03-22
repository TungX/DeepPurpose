from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

# Load Data, an array of SMILES for drug, an array of Amino Acid Sequence for Target and an array of binding values/0-1 label.
# e.g. ['Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1', ...], ['MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH...', ...], [0.46, 0.49, ...]
# In this example, BindingDB with Kd binding score is used.
import pandas as pd
df_ic = pd.read_csv('/data/PANCANCER_IC.csv')

df_ic = df_ic[['Drug name', 'Cell line name', 'IC50']]
df_ic = df_ic.rename(columns={'Drug name': 'drug', 'Cell line name': 'cell_line', 'IC50': 'ic50'})
df_drug_smile = pd.read_csv('/data/drug_smiles.csv')
df_drug_smile = df_drug_smile[['name', 'CanonicalSMILES']]
df_drug_smile = df_drug_smile.rename(columns={'name': 'drug', 'CanonicalSMILES': 'smiles'})
df = pd.merge(df_ic, df_drug_smile)
df_cell_line_feature = pd.read_csv('/data/PANCANCER_Genetic_feature.csv')
df_cell_line_feature = df_cell_line_feature[df_cell_line_feature.is_mutated == 1]
df_cell_line_feature = df_cell_line_feature[['cell_line_name', 'genetic_feature']]
df_cell_line_feature = df_cell_line_feature.rename(columns={'cell_line_name': 'cell_line', 'genetic_feature': 'feature'})

cell_line_features = {}

for index, row in df_cell_line_feature.iterrows():
	cell_line = row['cell_line']
	if row['cell_line'] not in cell_line_features:
		cell_line_features[cell_line] = []
	cell_line_features[cell_line].append(row['feature'])

features = []
for index, row in df.iterrows():
	cell_line = row['cell_line']
	if cell_line in cell_line_features:
		features.append(','.join(cell_line_features[cell_line]))
	else:
		features.append('')

df['features'] = features

X_drug = df['smiles']
X_cell_line = df['features']

y = df['ic50']
# X_drug, X_target, y  = process_BindingDB(path='/data/BindingDB_All.tsv',
# 					 y = 'Kd', 
# 					 binary = False, 
# 					 convert_to_log = True)

# # Type in the encoding names for drug/protein.
drug_encoding, target_encoding = 'MPNN', 'Transformer'
drug_encoding, target_encoding = 'MPNN', 'CNN'
drug_encoding, target_encoding = 'MPNN', 'RNN'
drug_encoding, target_encoding = 'GIN', 'Transformer'
drug_encoding, target_encoding = 'GIN', 'CNN'
drug_encoding, target_encoding = 'GIN', 'RNN'
# data_using_cell_line_process
# Data processing, here we select cold protein split setup.
train, val, test = data_using_cell_line_process(X_drug, X_cell_line, y, 
                                drug_encoding, target_encoding, 
                                split_method='random', 
                                frac=[0.7,0.1,0.2])

# print(test['target_encoding'].values[0])
print(test.keys())
for index, row in test.iterrows():
	print(row)
	break

# Generate new model using default parameters; also allow model tuning via input parameters.
config = generate_config(drug_encoding, target_encoding, transformer_n_layer_target = 8, train_epoch = 10)
net = models.model_initialize(**config)

# Train the new model.
# Detailed output including a tidy table storing validation loss, metrics, AUC curves figures and etc. are stored in the ./result folder.

# try:
# 	net.train(train, val, test)
# except Exception as e:
#     print(e)

# net.save_model('/models/dit_drug_cell_line_10_final')
# # or simply load pretrained model from a model directory path or reproduced model name such as DeepDTA
# net = models.model_pretrained(MODEL_PATH_DIR or MODEL_NAME)

# # Repurpose using the trained model or pre-trained model
# # In this example, loading repurposing dataset using Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
# X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
# target, target_name = load_SARS_CoV_Protease_3CL()

# _ = models.repurpose(X_repurpose, target, net, drug_name, target_name)

# # Virtual screening using the trained model or pre-trained model 
# X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

# _ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)
