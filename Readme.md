Six steps to replicate the results:

- Step 1. Install rlberry. Please follow the guideline at https://rlberry.readthedocs.io/en/latest/installation.html

- Step 2. Open a terminal and change directory to the directory ReplicateExperiments.

- Step 3. Activate rlberry environment with command.

```bash
$ conda activate rlberry
```

- Step 4. Train and test four agents: the optimal non-stationary agent, the AOMultiRL agent with a given distinguishing set, the one-episode UCBVI agent and the random agent. 

```bash
$ python AOMultiRL1.py
```
At the end of this command, results for these four agents are saved in the directory `Data/AOMultiRL1`.
- Step 5. Train and test the AOMultiRL2 agent that discovers a distinguishing set on its own.
```bash
$ python AOMultiRL2.py
```
At the end of this command, results for these four agents are saved in the directory `Data/AOMultiRL2`.
- Step 6. Visualize the results by running
```bash
$ utils.py 
```

