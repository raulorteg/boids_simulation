# Boids simulation

_Note: this is simply a for fun mini-project_


![](figures/animation_noforces.gif)

## Installation

_Note: Developed with Python 3.12.3_

1. Create virtual environment
```bash
python3 -m venv ~/python-envs/birdflock
```
2. Activate virtual environment
```bash
source ~/python-envs/birdflock/bin/activate
```
3. Install requirements
```bash
python3 -m pip install -r requirements.txt
```
## Usage

Run the script with the following command:  

```bash
python simulate.py [options]
```

## Options:

| **Option**           | **Type**       | **Description**                                                                                          | **Default**           |
|----------------------|----------------|----------------------------------------------------------------------------------------------------------|-----------------------|
| `--population`       | *int*          | The population size of the boids.                                                                         | 10                    |
| `--cutoff_att`       | *float*        | The attraction cutoff distance.                                                                            | 3.0                   |
| `--cutoff_rep`       | *float*        | The repulsion cutoff distance.                                                                            | 1.0                   |
| `--cohesion_factor`  | *float*        | The factor controlling how much boids are pulled towards the center of the group.                         | 0.5                   |
| `--alignment_factor` | *float*        | The factor controlling how much boids align with the direction of their neighbors.                        | 0.2                   |
| `--separation_factor`| *float*        | The factor controlling how much boids avoid crowding together.                                            | 5.0                   |
| `--repulsion_factor` | *float*        | The factor controlling the repulsion strength between agents.                                             | 1.0                   |
| `--box_size`         | *float*        | The size of the simulation box.                                                                           | 20.0                  |
| `--max_steps`        | *int*          | The maximum number of iterations for the simulation.                                                     | 400                   |
| `--dt`               | *float*        | The delta time for each step in the simulation.                                                           | 0.04                  |
| `--fps`              | *float*        | The frames per second for the final visualization.                                                        | 60                    |
| `--show_forces`        | *flag*           | If provided, the visualization will include forces.                                           | Not included          |

### Examples

1. **Run with a simulation with 100 boids in a box 200 by 200:**
   ```bash
   python simulate.py --population=100 --box_size=100
   ```


## Gallery
![](figures/animation_forces.gif)