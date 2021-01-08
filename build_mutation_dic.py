import pandas as pd
# reader = pd.read_csv('/data/PANCANCER_Genetic_feature.csv')
# values = reader.genetic_feature.values

# print(len(values))
# mutations = {}

# for x in values:
#     mutations[x] = 1

# with open('./DeepPurpose/ESPF/cell_line_mut.csv', 'w') as the_file:
#     the_file.write('genetic_feature\n')
#     for x in mutations:
#         the_file.write(x+'\n')
mut_csv = pd.read_csv('./DeepPurpose/ESPF/cell_line_mut.csv')
mut_name = mut_csv['genetic_feature'].values
mut_name2idmut_p = dict(zip(mut_name, range(0, len(mut_name))))
print(mut_name2idmut_p)