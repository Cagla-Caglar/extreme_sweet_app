# ============================================================
# REGRESSION ANALYSIS: DESCRIPTOR AND FINGERPRINT COMPUTATION
# ============================================================

# %%
from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np

np.seterr(over='raise', invalid='raise')

calc = Calculator(descriptors, ignore_3D=True)

file_path = "C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx"
data = pd.read_excel(file_path)

molecule_names = data['Name']
smiles_codes = data['Smiles']
log_sweetness_values = data['logSw']

molecule_data = dict(zip(molecule_names, smiles_codes))
sweetness_values = log_sweetness_values.tolist()

results = []
errors = []

def handle_large_values(x):
    """Replace inf and extreme values with NaN."""
    try:
        if isinstance(x, (int, float)) and (np.isinf(x) or abs(x) > 1e10):
            return np.nan
        return x
    except (OverflowError, TypeError, ValueError):
        return np.nan

for name, smiles in molecule_data.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            result = calc(mol)
            desc_dict = result.fill_missing().asdict()
            desc_dict = {k: handle_large_values(v) for k, v in desc_dict.items()}
            results.append(desc_dict)
        except Exception as e:
            errors.append((name, str(e)))

df = pd.DataFrame(results)
df = df.astype(np.float64)
df = df.apply(lambda col: col.apply(handle_large_values))
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns containing any NaN values
df.dropna(axis=1, how='any', inplace=True)

df['Molecule'] = molecule_names
df['Sweetness'] = sweetness_values
df = df[['Molecule'] + [col for col in df.columns if col not in ['Molecule', 'Sweetness']] + ['Sweetness']]

output_file_path = "C:\\Users\\pc\\Desktop\\sweetenersdb_molecular_descriptors_with_sweetness_intensities.csv"
df.to_csv(output_file_path, index=False)

# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx'
df = pd.read_excel(file_path)

# Extract molecule names and SMILES
molecule_names = df['Name']
smiles_list = df['Smiles']

# Function to calculate Morgan fingerprints
def calculate_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        return list(fingerprint)
    else:
        return [None] * nBits

# Calculate Morgan fingerprints
fingerprints = [calculate_morgan_fingerprint(smiles) for smiles in smiles_list]

# Create a DataFrame with the results
fingerprints_df = pd.DataFrame(fingerprints)
fingerprints_df.columns = [f'Bit_{i}' for i in range(2048)]
fingerprints_df.insert(0, 'Molecule', molecule_names)

# Save the DataFrame to a CSV file
output_file_path = 'C:\\Users\\pc\\Desktop\\sweetenersdb_morgan_fingerprints.csv'
fingerprints_df.to_csv(output_file_path, index=False)

print(f'Morgan fingerprints saved to {output_file_path}')
# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx'
df = pd.read_excel(file_path)

# Extract molecule names and SMILES
molecule_names = df['Name']
smiles_list = df['Smiles']

# Function to calculate MACCS keys
def calculate_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = MACCSkeys.GenMACCSKeys(mol)
        return list(map(int, fingerprint)) 
    else:
        return [None] * 167  

# Calculate MACCS keys fingerprints for each molecule
fingerprints_maccs = [calculate_maccs(smiles) for smiles in smiles_list]

# Create a DataFrame with the results
columns_maccs = ['Bit_{}'.format(i) for i in range(167)]
df_maccs = pd.DataFrame(fingerprints_maccs, columns=columns_maccs)

# Insert molecule names
df_maccs.insert(0, 'Molecule', molecule_names)

# Save the DataFrame to a CSV file
output_file_path = 'C:\\Users\\pc\\Desktop\\sweetenersdb_maccs_fingerprints.csv'
df_maccs.to_csv(output_file_path, index=False)

print(f'MACCS fingerprints saved to {output_file_path}')

# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx'
df = pd.read_excel(file_path)

# Extract molecule names and SMILES
molecule_names = df['Name']
smiles_list = df['Smiles']

# Function to calculate Atom Pair fingerprints
def calculate_atom_pair(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return list(map(int, fingerprint))  
    else:
        return [0] * n_bits  

# Calculate Atom Pair fingerprints for each molecule
fingerprints_atom_pair = [calculate_atom_pair(smiles) for smiles in smiles_list]

# Create a DataFrame with the results
columns_atom_pair = ['Bit_{}'.format(i) for i in range(2048)]
df_atom_pair = pd.DataFrame(fingerprints_atom_pair, columns=columns_atom_pair)

# Insert molecule names
df_atom_pair.insert(0, 'Molecule', molecule_names)

# Save the DataFrame to a CSV file
output_file_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_atom_pair_fingerprints.csv'
df_atom_pair.to_csv(output_file_path, index=False)

print(f'Atom Pair fingerprints saved to {output_file_path}')

# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx'
df = pd.read_excel(file_path)

# Extract molecule names and SMILES
molecule_names = df['Name']
smiles_list = df['Smiles']

# Function to calculate Topological Torsion fingerprints
def calculate_topological_torsion(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        return list(map(int, fingerprint))  
    else:
        return [0] * n_bits  

# Calculate Topological Torsion fingerprints for each molecule
fingerprints_topological_torsion = [calculate_topological_torsion(smiles) for smiles in smiles_list]

# Create a DataFrame with the results
columns_topological_torsion = ['Bit_{}'.format(i) for i in range(2048)]
df_topological_torsion = pd.DataFrame(fingerprints_topological_torsion, columns=columns_topological_torsion)

# Insert molecule names
df_topological_torsion.insert(0, 'Molecule', molecule_names)

# Save the DataFrame to a CSV file
output_file_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_topological_torsion_fingerprints.csv'
df_topological_torsion.to_csv(output_file_path, index=False)

print(f'Topological Torsion fingerprints saved to {output_file_path}')

# %%
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S2.xlsx'
df = pd.read_excel(file_path)

# Extract molecule names and SMILES
molecule_names = df['Name']
smiles_list = df['Smiles']

# Function to calculate Avalon fingerprints
def calculate_avalon(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        return list(map(int, fingerprint))  
    else:
        return [0] * n_bits 

# Calculate Avalon fingerprints for each molecule
fingerprints_avalon = [calculate_avalon(smiles) for smiles in smiles_list]

# Create a DataFrame with the results
columns_avalon = ['Bit_{}'.format(i) for i in range(2048)]
df_avalon = pd.DataFrame(fingerprints_avalon, columns=columns_avalon)

# Insert molecule names
df_avalon.insert(0, 'Molecule', molecule_names)

# Save the DataFrame to a CSV file
output_file_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_avalon_fingerprints.csv'
df_avalon.to_csv(output_file_path, index=False)

print(f'Avalon fingerprints saved to {output_file_path}')

# %%
import pandas as pd

# Load the datasets
molecular_descriptors = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_molecular_descriptors_with_sweetness_intensities.csv')
morgan_fingerprints = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_morgan_fingerprints.csv')
maccs_fingerprints = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_maccs_fingerprints.csv')
atom_pair_fingerprints = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_atom_pair_fingerprints.csv')
topological_torsion_fingerprints = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_topological_torsion_fingerprints.csv')
avalon_fingerprints = pd.read_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweetenersdb_avalon_fingerprints.csv')

# Extract the compound names and sweetness values
compound_names = molecular_descriptors.iloc[:, 0]
sweetness_values = molecular_descriptors.iloc[:, -1]


molecular_descriptors = molecular_descriptors.drop(columns=[molecular_descriptors.columns[0], molecular_descriptors.columns[-1]])
morgan_fingerprints = morgan_fingerprints.drop(columns=['Molecule'])
maccs_fingerprints = maccs_fingerprints.drop(columns=['Molecule'])
atom_pair_fingerprints = atom_pair_fingerprints.drop(columns=['Molecule'])
topological_torsion_fingerprints = topological_torsion_fingerprints.drop(columns=['Molecule'])
avalon_fingerprints = avalon_fingerprints.drop(columns=['Molecule'])

morgan_fingerprints.columns = ['ECFP4_' + col for col in morgan_fingerprints.columns]  
maccs_fingerprints.columns = ['MACCS_' + col for col in maccs_fingerprints.columns]
atom_pair_fingerprints.columns = ['AtomPair_' + col for col in atom_pair_fingerprints.columns]
topological_torsion_fingerprints.columns = ['TopoTorsion_' + col for col in topological_torsion_fingerprints.columns]
avalon_fingerprints.columns = ['Avalon_' + col for col in avalon_fingerprints.columns]

# Combine all datasets
combined_data = pd.concat([
    compound_names,
    molecular_descriptors,
    morgan_fingerprints,
    maccs_fingerprints,
    atom_pair_fingerprints,
    topological_torsion_fingerprints,
    avalon_fingerprints,
    sweetness_values
], axis=1)

# Save the combined dataset to a CSV file
combined_data.to_csv('C:\\Users\\pc\\Desktop\\thesis_project\\sweeteners_with_molecular_descriptors_and_fingerprints.csv', index=False)

# Display the shape of the combined dataset
print("Shape of the combined dataset:", combined_data.shape)



# ============================================================
# CLASSIFICATION ANALYSIS: DESCRIPTOR AND FINGERPRINT COMPUTATION
# ============================================================

# %%
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\ChemTastesDB\\foods-2444588 - Table S1.xlsx'
data = pd.read_excel(file_path)

# Directory to save the .mol files 
output_dir = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'
os.makedirs(output_dir, exist_ok=True)

# Function to generate 3D conformer and save as .mol file
def generate_3d_conformer(smiles, compound_number):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES {smiles} could not be parsed for compound {compound_number}.")
            return
        mol = Chem.AddHs(mol)
        
        # Generate conformers 
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        params.maxAttempts = 1000  
        params.randomSeed = 1
        conformers = AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params)
        
        if not conformers:
            print(f"Embedding failed for compound {compound_number}.")
            return

        # Optimize each conformer with MMFF and find the one with the lowest energy
        lowest_energy = float('inf')
        best_conformer = None
        for conf_id in conformers:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
            if mmff_props is None:
                continue
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94', confId=conf_id)
            energy = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id).CalcEnergy()
            if energy < lowest_energy:
                lowest_energy = energy
                best_conformer = conf_id
        
        if best_conformer is None:
            print(f"Optimization failed for compound {compound_number}.")
            return

        # Save the molecule file using only the compound number
        mol_file = os.path.join(output_dir, f"{compound_number}.mol")
        Chem.MolToMolFile(mol, mol_file, confId=best_conformer)
    except Exception as e:
        print(f"Error processing compound {compound_number}: {e}")

# Generate 3D conformers
for idx, row in data.iterrows():
    smiles = row['SMILES']
    compound_number = row['Number']
    generate_3d_conformer(smiles, compound_number)

print("3D conformers have been generated and saved.")



# %%
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors, is_missing

np.seterr(over='raise', invalid='raise')
calc = Calculator(descriptors)

# 3D descriptor names from Mordred
threed_desc_names = set()
for desc in descriptors.all:
    if hasattr(desc, 'require_3D') and desc.require_3D:
        threed_desc_names.add(str(desc))
calc_3d_check = Calculator(descriptors, ignore_3D=True)
names_2d = set(str(d) for d in calc_3d_check.descriptors)

mol_directory = r"C:\Users\pc\Desktop\sweet_nonsweet_mmff_conformers_moll"
file_path = r"C:\Users\pc\Desktop\ChemTastesDB\foods-2444588 - Table S1.xlsx"
data = pd.read_excel(file_path)

molecule_data = dict(zip(data['Number'], data['SMILES']))
class_values = dict(zip(data['Number'], data['Class']))
name_values = dict(zip(data['Number'], data['Name']))

def handle_large_values(x):
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            xv = float(x)
            if np.isinf(xv) or abs(xv) > 1e10:
                return np.nan
            return xv
        return x
    except (OverflowError, TypeError, ValueError):
        return np.nan

def read_mol_file(file_path):
    try:
        mol = Chem.MolFromMolFile(file_path)
        if mol:
            Chem.SanitizeMol(mol)
            return mol
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return None

results = []
processed_molecule_numbers = []

for number, smiles in molecule_data.items():
    mol_file_path = os.path.join(mol_directory, f"{number}.mol")
    mol = read_mol_file(mol_file_path)
    if mol:
        try:
            result = calc(mol)
            desc_dict = result.fill_missing().asdict()
            desc_dict = {k: handle_large_values(v) for k, v in desc_dict.items()}
            results.append(desc_dict)
            processed_molecule_numbers.append(number)
        except Exception as e:
            print(f"Error processing {number}_{name_values[number]}: {e}")

df = pd.DataFrame(results)

# Remove 3D descriptors
cols_to_drop = [c for c in df.columns if c not in names_2d]
print(f"Dropping {len(cols_to_drop)} 3D descriptor columns")
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

df = df.astype(np.float64)
df = df.apply(lambda col: col.apply(handle_large_values))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, how='any', inplace=True)

df.insert(0, 'Name', [f"{n}_{name_values[n]}" for n in processed_molecule_numbers])
df['Class'] = [1 if class_values[n] == 'Sweet' else 0 for n in processed_molecule_numbers]

print(f"Shape: {df.shape}")
print(f"Molecules: {len(processed_molecule_numbers)}")
print(f"Descriptors: {df.shape[1] - 2}")
df.to_csv("sweet_nonsweet_molecular_descriptors_with_taste_classes.csv", index=False)
print("Done.")


# %%
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import unidecode

# Path to the folder
folder_path = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'

def normalize_name(name):
    return unidecode.unidecode(name)

results = []
# List to store errors
errors = []

file_names = [f for f in os.listdir(folder_path) if f.endswith('.mol')]
file_names.sort(key=lambda x: int(x.split('_')[0]))

for file_name in file_names:
    if file_name.endswith('.mol'):
        try:
            # Normalize the compound name
            normalized_name = normalize_name(file_name)
            
            # Rename the file with the normalized name
            original_path = os.path.join(folder_path, file_name)
            normalized_path = os.path.join(folder_path, normalized_name)
            os.rename(original_path, normalized_path)
            
            # Extract the compound name from the normalized file name
            compound_name = normalized_name.split('_', 1)[1].rsplit('.', 1)[0]
            
            # Read the .mol file
            mol = Chem.MolFromMolFile(normalized_path)
            
            if mol is not None:
                # Calculate Morgan fingerprints
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprint_array = list(fingerprint)
                
                # Add the result to the list
                results.append([compound_name] + fingerprint_array)
            else:
                errors.append((normalized_name, 'Molecule could not be read'))
        except Exception as e:
            errors.append((file_name, str(e)))

# Convert the results to a DataFrame
columns = ['Compound_Name'] + [f'Bit_{i}' for i in range(2048)]
df = pd.DataFrame(results, columns=columns)

# Save the DataFrame to a CSV file
output_path = 'sweet_nonsweet_molecules_morgan_fingerprints.csv'
df.to_csv(output_path, index=False)

# Save the errors to a text file
error_path = 'errors.txt'
with open(error_path, 'w') as error_file:
    for error in errors:
        error_file.write(f'{error[0]}: {error[1]}\n')

print(f'Morgan fingerprints have been saved to {output_path}')
print(f'Errors have been logged to {error_path}')

# %%
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import unidecode

# Path to the folder 
folder_path = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'

# Function to normalize file names
def normalize_name(name):
    return unidecode.unidecode(name)

# List to store the results
results = []
# List to store errors
errors = []

file_names = [f for f in os.listdir(folder_path) if f.endswith('.mol')]
file_names.sort(key=lambda x: int(x.split('_')[0]))


for file_name in file_names:
    if file_name.endswith('.mol'):
        try:
            # Normalize the compound name
            normalized_name = normalize_name(file_name)
            
            # Rename the file with the normalized name
            original_path = os.path.join(folder_path, file_name)
            normalized_path = os.path.join(folder_path, normalized_name)
            os.rename(original_path, normalized_path)
            
            # Extract the compound name from the normalized file name
            compound_name = normalized_name.split('_', 1)[1].rsplit('.', 1)[0]
            
            # Read the .mol file
            mol = Chem.MolFromMolFile(normalized_path)
            
            if mol is not None:
                # Calculate MACCS fingerprints
                fingerprint = MACCSkeys.GenMACCSKeys(mol)
                fingerprint_array = list(fingerprint)
                
                # Add the result to the list
                results.append([compound_name] + fingerprint_array)
            else:
                errors.append((normalized_name, 'Molecule could not be read'))
        except Exception as e:
            errors.append((file_name, str(e)))

# Convert the results to a DataFrame
columns = ['Compound_Name'] + [f'Bit_{i}' for i in range(167)]  
df = pd.DataFrame(results, columns=columns)

# Save the DataFrame to a CSV file
output_path = 'sweet_nonsweet_molecules_maccs_fingerprints.csv'
df.to_csv(output_path, index=False)

# Save the errors to a text file
error_path = 'errors.txt'
with open(error_path, 'w') as error_file:
    for error in errors:
        error_file.write(f'{error[0]}: {error[1]}\n')

print(f'MACCS fingerprints have been saved to {output_path}')
print(f'Errors have been logged to {error_path}')


# %%
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Path to the folder 
folder_path = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'

# List to store the results
results = []
# List to store errors
errors = []

file_names = [f for f in os.listdir(folder_path) if f.endswith('.mol')]
file_names.sort(key=lambda x: int(x.split('_')[0]))

# Function to calculate Atom Pair fingerprints
def calculate_atom_pair(mol, n_bits=2048):
    if mol is not None:
        fingerprint = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return list(map(int, fingerprint)) 
    else:
        return [0] * n_bits  

for file_name in file_names:
    try:
        # Extract the compound name from the file name
        compound_name = file_name.split('_', 1)[1].rsplit('.', 1)[0]
        
        # Read the .mol file
        mol_path = os.path.join(folder_path, file_name)
        mol = Chem.MolFromMolFile(mol_path)
        
        if mol is not None:
            # Calculate Atom Pair fingerprints
            fingerprint_array = calculate_atom_pair(mol)
            
            # Add the result to the list
            results.append([compound_name] + fingerprint_array)
        else:
            errors.append((file_name, 'Molecule could not be read'))
    except Exception as e:
        errors.append((file_name, str(e)))

# Convert the results to a DataFrame
if results:
    columns = ['Compound_Name'] + [f'Bit_{i}' for i in range(len(results[0]) - 1)]  
    df = pd.DataFrame(results, columns=columns)

    # Save the DataFrame to a CSV file
    output_path = 'sweet_nonsweet_molecules_atom_pair_fingerprints.csv'
    df.to_csv(output_path, index=False)

    print(f'Atom Pair fingerprints have been saved to {output_path}')
else:
    print('No valid fingerprints were calculated.')

# Save the errors to a text file
error_path = 'errors.txt'
with open(error_path, 'w') as error_file:
    for error in errors:
        error_file.write(f'{error[0]}: {error[1]}\n')

print(f'Errors have been logged to {error_path}')


# %%
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AtomPairs import Torsions

# Path to the folder 
folder_path = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'

# List to store the results
results = []
# List to store errors
errors = []

file_names = [f for f in os.listdir(folder_path) if f.endswith('.mol')]
file_names.sort(key=lambda x: int(x.split('_')[0]))

for file_name in file_names:
    try:
        # Extract the compound name from the file name
        compound_name = file_name.split('_', 1)[1].rsplit('.', 1)[0]
        
        # Read the .mol file
        mol_path = os.path.join(folder_path, file_name)
        mol = Chem.MolFromMolFile(mol_path)
        
        if mol is not None:
            # Calculate Topological Torsion fingerprints
            fingerprint = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
            fingerprint_array = list(fingerprint.GetNonzeroElements().keys())
            
            # Add the result to the list
            results.append([compound_name] + fingerprint_array)
        else:
            errors.append((file_name, 'Molecule could not be read'))
    except Exception as e:
        errors.append((file_name, str(e)))

# Convert the results to a DataFrame
if results:
    max_length = max(len(f) for f in results) - 1
    columns = ['Compound_Name'] + [f'Bit_{i}' for i in range(max_length)]  
    for result in results:
        result.extend([0] * (max_length - (len(result) - 1)))  
    df = pd.DataFrame(results, columns=columns)

    # Save the DataFrame to a CSV file
    output_path = 'sweet_nonsweet_molecules_topological_torsion_fingerprints.csv'
    df.to_csv(output_path, index=False)

    print(f'Topological Torsion fingerprints have been saved to {output_path}')
else:
    print('No valid fingerprints were calculated.')

# Save the errors to a text file
error_path = 'errors.txt'
with open(error_path, 'w') as error_file:
    for error in errors:
        error_file.write(f'{error[0]}: {error[1]}\n')

print(f'Errors have been logged to {error_path}')

# %%
import os
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools

# Path to the folder containing the .mol files
folder_path = 'C:/Users/pc/Desktop/sweet_nonsweet_mmff_conformers_mol'

results = []
errors = []

file_names = [f for f in os.listdir(folder_path) if f.endswith('.mol')]
file_names.sort(key=lambda x: int(x.split('_')[0]))

# Function to calculate Avalon fingerprints from a molecule
def calculate_avalon(mol, n_bits=2048):
    if mol is not None:
        fingerprint = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        return list(map(int, fingerprint))  
    else:
        return [0] * n_bits  

for file_name in file_names:
    try:
        # Extract the compound name from the file name
        compound_name = file_name.split('_', 1)[1].rsplit('.', 1)[0]
        
        # Read the .mol file
        mol_path = os.path.join(folder_path, file_name)
        mol = Chem.MolFromMolFile(mol_path)
        
        if mol is not None:
            # Calculate Avalon fingerprints
            fingerprint_array = calculate_avalon(mol)
            
            # Add the result to the list
            results.append([compound_name] + fingerprint_array)
        else:
            errors.append((file_name, 'Molecule could not be read'))
    except Exception as e:
        errors.append((file_name, str(e)))

# Convert the results to a DataFrame
if results:
    columns = ['Compound_Name'] + [f'Bit_{i}' for i in range(len(results[0]) - 1)]  
    df = pd.DataFrame(results, columns=columns)

    # Save the DataFrame to a CSV file
    output_path = 'sweet_nonsweet_molecules_avalon_fingerprints.csv'
    df.to_csv(output_path, index=False)

    print(f'Avalon fingerprints have been saved to {output_path}')
else:
    print('No valid fingerprints were calculated.')

# Save the errors to a text file
error_path = 'errors.txt'
with open(error_path, 'w') as error_file:
    for error in errors:
        error_file.write(f'{error[0]}: {error[1]}\n')

print(f'Errors have been logged to {error_path}')


# %%
import pandas as pd

# Load the datasets
molecular_descriptors = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecular_descriptors_with_taste_classes.csv')
morgan_fingerprints = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_morgan_fingerprints.csv')
maccs_fingerprints = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_maccs_fingerprints.csv')
atom_pair_fingerprints = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_atom_pair_fingerprints.csv')
topological_torsion_fingerprints = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_topological_torsion_fingerprints.csv')
avalon_fingerprints = pd.read_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_avalon_fingerprints.csv')

# Extract the compound names and taste classes
compound_names = molecular_descriptors.iloc[:, 0]
taste_classes = molecular_descriptors.iloc[:, -1]

molecular_descriptors = molecular_descriptors.drop(columns=[molecular_descriptors.columns[0], molecular_descriptors.columns[-1]])
morgan_fingerprints = morgan_fingerprints.drop(columns=['Compound_Name'])
maccs_fingerprints = maccs_fingerprints.drop(columns=['Compound_Name'])
atom_pair_fingerprints = atom_pair_fingerprints.drop(columns=['Compound_Name'])
topological_torsion_fingerprints = topological_torsion_fingerprints.drop(columns=['Compound_Name'])
avalon_fingerprints = avalon_fingerprints.drop(columns=['Compound_Name'])

morgan_fingerprints.columns = ['ECFP4_' + col for col in morgan_fingerprints.columns]  
maccs_fingerprints.columns = ['MACCS_' + col for col in maccs_fingerprints.columns]
atom_pair_fingerprints.columns = ['AtomPair_' + col for col in atom_pair_fingerprints.columns]
topological_torsion_fingerprints.columns = ['TopoTorsion_' + col for col in topological_torsion_fingerprints.columns]
avalon_fingerprints.columns = ['Avalon_' + col for col in avalon_fingerprints.columns]


# Combine all datasets
combined_data = pd.concat([
    compound_names,
    molecular_descriptors,
    morgan_fingerprints,
    maccs_fingerprints,
    atom_pair_fingerprints,
    topological_torsion_fingerprints,
    avalon_fingerprints,
    taste_classes
], axis=1)

# Save the combined dataset to a CSV file
combined_data.to_csv(r'C:\Users\pc\Desktop\thesis_project\sweet_nonsweet_molecules_with_molecular_descriptors_and_fingerprints.csv', index=False)

# Display the shape of the combined dataset
print("Shape of the combined dataset:", combined_data.shape)

# %%
# %%
import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweet_nonsweet_molecules_with_molecular_descriptors_and_fingerprints.csv'
data = pd.read_csv(file_path)

# Encode the 'Class' column
data['Class'] = data['Class'].map({'Sweet': 1, 'Non-sweet': 0})

# Save the updated dataset
output_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweet_nonsweet_molecules_with_molecular_descriptors_and_fingerprints_dataset.csv'
data.to_csv(output_path, index=False)


# %%
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweet_nonsweet_molecules_with_molecular_descriptors_and_fingerprints_dataset.csv'
dataset = pd.read_csv(file_path)

# Separate features and target
X = dataset.drop(columns=['Class'])
y = dataset['Class']

# Split the dataset into 70% training and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Combine the features and target for both training and test sets
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Save the split datasets 
train_set_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweet_nonsweet_train_set.csv'
test_set_path = 'C:\\Users\\pc\\Desktop\\thesis_project\\sweet_nonsweet_test_set.csv'
train_set.to_csv(train_set_path, index=False)
test_set.to_csv(test_set_path, index=False)

train_set_path, test_set_path

