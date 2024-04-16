#!/usr/bin/env python
# coding: utf-8

from docking_functions import smi_txt_to_pdb, set_element, pdb_to_pdbqt, make_autodock_vina_config, autodock_vina
import pandas as pd
import uuid
from parsl import python_app, bash_app
from parsl.data_provider.files import File as PFile
from concurrent.futures import as_completed
from time import monotonic
from ml_functions import train_model, run_model

smi_file_name_ligand = 'dataset_orz_original_1k.csv'

search_space = pd.read_csv(smi_file_name_ligand)
search_space = search_space[['TITLE','SMILES']]

print(search_space.head(5))


# We define new versions of the functions above and annotate them as Parsl apps. To help Parsl track then flow of data between apps we add a new argument "outputs". This is used by Parsl to track the files that are produced by an app such that they can be passed to subsequent apps.

receptor = '1iep_receptor.pdbqt'
ligand = 'paxalovid-molecule-coords.pdbqt'


@python_app
def parsl_smi_to_pdb(smiles, outputs=[], inputs=[]):
    from rdkit import Chem
    from rdkit.Chem import AllChem
   
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    
    writer = Chem.PDBWriter(outputs[0].filepath)
    writer.write(mol)
    writer.close()
    
    return True

@bash_app
def parsl_set_element(f, outputs=[], inputs=[], stdout="my_stdout"):
   
    tcl_script = "set_element.tcl"
    command = (
        f"vmd -dispdev text -e {tcl_script} -args {f} {outputs[0]}"
    )
    return command

@bash_app
def parsl_pdb_to_pdbqt(f, outputs=[], ligand = True, inputs=[]):
    import os
    from pathlib import Path
    autodocktools_path = os.getenv('MGLTOOLS_HOME') 

    # Select the correct settings for ligand or receptor preparation
    script, flag = (
        ("prepare_ligand4.py", "l") if ligand else ("prepare_receptor4.py", "r")
    )

    command = (
        f"{'python2.7'}"
        f" {Path(autodocktools_path) / 'MGLToolsPckgs/AutoDockTools/Utilities24' / script}"
        f" -{flag} {f}"
        f" -o {outputs[0]}"
        f" -U nphs_lps_waters"
    )
    return command

@python_app
def parsl_make_autodock_config(
    r,
    l,
    output_pdbqt,
    outputs=[], 
    center=(15.614, 53.380, 15.455), size=(20, 20, 20),
    exhaustiveness=1, num_modes= 20, energy_range = 10,
    inputs=[]):
    
    import os
    # Format configuration file
    file_contents = (
        f"receptor = {r}\n"
        f"ligand = {l}\n"
        f"center_x = {center[0]}\n"
        f"center_y = {center[1]}\n"
        f"center_z = {center[2]}\n"
        f"size_x = {size[0]}\n"
        f"size_y = {size[1]}\n"
        f"size_z = {size[2]}\n"
        f"exhaustiveness = {exhaustiveness}\n"
        f"num_modes = {num_modes}\n"
        f"energy_range = {energy_range}\n"
        f"out = {output_pdbqt}\n"
        #f"log = {output_log_file}\n"
    )
    # Write configuration file
    with open(outputs[0].local_path, "w") as f:
        f.write(file_contents)
        
    return True
    
@python_app
def parsl_autodock_vina(smiles, f, num_cpu = 1, inputs=[], outputs=[], parsl_resource_specification={'cores':4}):
    import subprocess

    autodock_vina_exe = "vina"
    try:
        command = f"{autodock_vina_exe} --config {f} --cpu {num_cpu} --log vinaoutput"
	#print(command)
        result = subprocess.check_output(command.split(), encoding="utf-8")

	# find the last row of the table and extract the affinity score
        result_list = result.split('\n')
        last_row = result_list[-3]
        score = last_row.split()
        return (smiles, float(score[1]))
    except subprocess.CalledProcessError as e:
        return (f"Command '{e.cmd}' returned non-zero exit status {e.returncode} {e.output}")
    except Exception as e:
        return (f"Error: {e}")

@python_app
def cleanup(dock_future, pdb, pdb_coords, pdb_qt, autodoc_config, docking):
    os.remove(pdb)
    os.remove(pdb_coords)
    os.remove(pdb_qt)
    os.remove(autodoc_config)
    os.remove(docking)


# #### Configure Parsl

from parsl.executors.taskvine import TaskVineExecutor
from parsl.executors.taskvine import TaskVineManagerConfig
from parsl.executors.taskvine import TaskVineFactoryConfig
from parsl.executors.taskvine.stub_staging_provider import StubStaging
from parsl.config import Config
import parsl

#manager_config=TaskVineManagerConfig(init_command='cp /home/scuzee/parsl-docking-tutorial/* . ; export MGLTOOLS_HOME=/home/scuzee/lsst/lsst_stack/conda/miniconda3-py38_4.9.2/envs/parsldockenv/ ;'),

config = Config(
    executors=[TaskVineExecutor(
	manager_config=TaskVineManagerConfig(init_command='export MGLTOOLS_HOME=$CONDA_PREFIX ;', port=9129, project_name="tv_parsl"),
	factory_config=TaskVineFactoryConfig(min_workers=100, max_workers=100, python_env="environment.tar.gz", batch_type="condor", workers_per_cycle=200, cores=12),
        storage_access=[StubStaging()],
    )]
)
parsl.clear()
parsl.load(config)

# Run some simulations

train_data = []
futures = []

set_element_tcl = PFile('set_element.tcl')
f_receptor = PFile(receptor)
while len(futures) < 5: 
    
    selected = search_space.sample(1).iloc[0]
    title, smiles = selected['TITLE'], selected['SMILES'] 
    
    # workflow
    fname = uuid.uuid4().hex
   
    f_pdb = PFile('taskvinetemp://%s.pdb' % fname)
    f_coords_pdb = PFile(f'taskvinetemp://{fname}-coords.pdb')
    f_coords_pdbqt = PFile(f'taskvinetemp://{fname}-coords.pdbqt')
    f_config = PFile('taskvinetemp://%s-config.txt' % fname)
    f_bringback = PFile(f"{fname}-out.pdb")
    #f_bigfile = PFile(f"taskvinetemp://{fname}-bigfile")
    
    #f_pdb = PFile('%s.pdb' % fname)
    #f_coords_pdb = PFile(f'{fname}-coords.pdb')
    #f_coords_pdbqt = PFile(f'{fname}-coords.pdbqt')
    #f_config = PFile('%s-config.txt' % fname)
    #f_bringback = PFile(f"{fname}-out.pdb")
    #f_bigfile = PFile(f"{fname}-bigfile")

    smi_future = parsl_smi_to_pdb(smiles, outputs=[f_pdb])
    element_future = parsl_set_element(smi_future.outputs[0], outputs=[f_coords_pdb], inputs=[set_element_tcl]) 
    pdbqt_future = parsl_pdb_to_pdbqt(element_future.outputs[0], outputs=[f_coords_pdbqt])
    config_future = parsl_make_autodock_config(f_receptor, pdbqt_future.outputs[0], '%s-out.pdb' % fname, outputs=[f_config])
    dock_future = parsl_autodock_vina(smiles, config_future.outputs[0], outputs=[f_bringback], inputs=[pdbqt_future.outputs[0], f_receptor], parsl_resource_specification={'cores':12})

    futures.append(dock_future)


while len(futures) > 0:
    future = next(as_completed(futures))
    print(future.result())
    smiles, score = future.result()
    futures.remove(future)

    print(f'Computation for {smiles} succeeded: {score}')
    
    train_data.append({
            'smiles': smiles,
            'score': score,
            'time': monotonic()
    })


#print("done")
#parsl.dfk().cleanup()
#exit()

# Train the model on those simulations
from ml_functions import train_model, run_model
training_df = pd.DataFrame(train_data)
m = train_model(training_df)
predictions = run_model(m, search_space['SMILES'])
predictions.sort_values('score', ascending=True).head(5)


# # Part 4: Putting it all together
# 
# We now combine the parallel ParslDock workflow with the machine learning algorithm in an iterative fashion. Here each round will 1) train a machine learning model based on previous simulations; 2) apply the machine learning model to all remaining molecules; 3) select the top predicted scores; 4) run simulations on the top molecules. 

futures = []
train_data = []
smiles_simulated = []
initial_count = 5
num_loops = 1
batch_size = 1000

print("Begin New Simulations")


# start with an initial set of random smiles
for i in range(initial_count):
    selected = search_space.sample(1).iloc[0]
    title, smiles = selected['TITLE'], selected['SMILES'] 

    # workflow
    fname = uuid.uuid4().hex
    
    f_pdb = PFile('taskvinetemp://%s.pdb' % fname)
    f_coords_pdb = PFile(f'taskvinetemp://{fname}-coords.pdb')
    f_coords_pdbqt = PFile(f'taskvinetemp://{fname}-coords.pdbqt')
    f_config = PFile('taskvinetemp://%s-config.txt' % fname)
    f_bringback = PFile(f"{fname}-out.pdb")
    #f_bigfile = PFile(f"taskvinetemp://{fname}-bigfile")
    
    #f_pdb = PFile('%s.pdb' % fname)
    #f_coords_pdb = PFile(f'{fname}-coords.pdb')
    #f_coords_pdbqt = PFile(f'{fname}-coords.pdbqt')
    #f_config = PFile('%s-config.txt' % fname)
    #f_bringback = PFile(f"{fname}-out.pdb")
    #f_bigfile = PFile(f"{fname}-bigfile")

    smi_future = parsl_smi_to_pdb(smiles, outputs=[f_pdb])
    element_future = parsl_set_element(smi_future.outputs[0], outputs=[f_coords_pdb], inputs=[set_element_tcl]) 
    pdbqt_future = parsl_pdb_to_pdbqt(element_future.outputs[0], outputs=[f_coords_pdbqt])
    config_future = parsl_make_autodock_config(f_receptor, pdbqt_future.outputs[0], '%s-out.pdb' % fname, outputs=[f_config])
    dock_future = parsl_autodock_vina(smiles, config_future.outputs[0], outputs=[f_bringback], inputs=[pdbqt_future.outputs[0], f_receptor], parsl_resource_specification={'cores':8})
    
    futures.append(dock_future)

# wait for all the futures to finish
while len(futures) > 0:
    future = next(as_completed(futures))
    smiles, score = future.result()
    futures.remove(future)

    print(f'Computation for {smiles} succeeded: {score}')
    
    train_data.append({
            'smiles': smiles,
            'score': score,
            'time': monotonic()
    })
    smiles_simulated.append(smiles)


# train model, run inference, and run more simulations
for i in range(num_loops):
    print(f"\nStarting batch {i}")
    m = train_model(training_df)
    predictions = run_model(m, search_space['SMILES'])
    predictions.sort_values('score', ascending=True, inplace=True) #.head(5)
    
    train_data = [] 
    futures = []
    batch_count = 0
    for smiles in predictions['smiles']:
        if smiles not in smiles_simulated:
            # workflow
            fname = uuid.uuid4().hex
            f_pdb = PFile('taskvinetemp://%s.pdb' % fname)
            f_coords_pdb = PFile(f'taskvinetemp://{fname}-coords.pdb')
            f_coords_pdbqt = PFile(f'taskvinetemp://{fname}-coords.pdbqt')
            f_config = PFile('taskvinetemp://%s-config.txt' % fname)
            f_bringback = PFile(f"{fname}-out.pdb")
            #f_bigfile = PFile(f"taskvinetemp://{fname}-bigfile")
            
            #f_pdb = PFile('%s.pdb' % fname)
            #f_coords_pdb = PFile(f'{fname}-coords.pdb')
            #f_coords_pdbqt = PFile(f'{fname}-coords.pdbqt')
            #f_config = PFile('%s-config.txt' % fname)
            #f_bringback = PFile(f"{fname}-out.pdb")
            #f_bigfile = PFile(f"{fname}-bigfile")

            smi_future = parsl_smi_to_pdb(smiles, outputs=[f_pdb])
            element_future = parsl_set_element(smi_future.outputs[0], outputs=[f_coords_pdb], inputs=[set_element_tcl]) 
            pdbqt_future = parsl_pdb_to_pdbqt(element_future.outputs[0], outputs=[f_coords_pdbqt])
            config_future = parsl_make_autodock_config(f_receptor, pdbqt_future.outputs[0], '%s-out.pdb' % fname, outputs=[f_config])
            dock_future = parsl_autodock_vina(smiles, config_future.outputs[0], outputs=[f_bringback], inputs=[pdbqt_future.outputs[0], f_receptor], parsl_resource_specification={'cores':12})
            
            futures.append(dock_future)
            batch_count += 1
            
        if batch_count > batch_size: 
            break

    # wait for all the workflows to complete
    while len(futures) > 0:
        future = next(as_completed(futures))
        smiles, score = future.result()
        futures.remove(future)

        print(f'Computation for {smiles} succeeded: {score}')

        train_data.append({
                'smiles': smiles,
                'score': score,
                'time': monotonic()
        })
        smiles_simulated.append(smiles)
   
                     
    training_df = pd.concat((training_df, pd.DataFrame(train_data)), ignore_index=True)

print("done")
parsl.dfk().cleanup()
exit()
