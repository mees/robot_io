## Data Structure after Preprocessing
Each interaction timestep is stored in a dictionary inside a numpy file `.npz` and contains all corresponding sensory observations, different action spaces, state information and language annoations. The data structure is the same as that supported in [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md). You can load each data file with the following python code:
```python
import numpy as np
filepath = "/tmp/episode_0000000.npz"
data = np.load(filepath, allow_pickle=True)
data = dict(data)
data.keys()
>>> dict_keys(['actions', 'rel_actions_world', 'rel_actions_gripper', 'robot_obs', 'rgb_gripper', 'depth_gripper', 'depth_static', 'rgb_static'])
```
### Camera Observations
The keys to access the different camera observations are:
```
['rgb_static'] (dtype=np.uint8, shape=(200, 200, 3)),
['rgb_gripper'] (dtype=np.uint8, shape=(84, 84, 3)),
['rgb_tactile'] (dtype=np.uint8, shape=(160, 120, 6)),
['depth_static'] (dtype=np.float32, shape=(200, 200)),
['depth_gripper'] (dtype=np.float32, shape=(84, 84)),
['depth_tactile'] (dtype=np.float32, shape=(160, 120, 2))
```
### Actions
Actions are in cartesian space and define the desired tcp pose wrt to the world frame and the binary gripper action.
The keys to access the 7-DOF absolute and relative actions are:
(tcp = tool center point, i.e. a virtual frame between the gripper finger tips of the robot)
```
['actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in absolute world coordinates
tcp orientation (3): euler angles x,y,z in absolute world coordinates
gripper_action (1): binary (close = -1, open = 1)

['rel_actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
gripper_action (1): binary (close = -1, open = 1)
```
For inference, Calvin env accepts both absolute and relative actions. To use absolute actions, the action is specified as a 3-tuple
`action = ((x,y,z), (euler_x, euler_y, euler_z), (gripper))`. To use relative actions, the action is specified as a
7-tuple `action = (x,y,z, euler_x, euler_y, euler_z, gripper)`. IMPORTANT: the environment expects the relative actions
to be scaled like the `rel_actions` in the dataset.

### State Observation
The keys to access the scene state information containing the position and orientation of all objects in the scenes
(we do not use them to better capture challenges present in real-world settings):
```
['scene_obs']
(dtype=np.float32, shape=(24,))
sliding door (1): joint state
drawer (1): joint state
button (1): joint state
switch (1): joint state
lightbulb (1): on=1, off=0
green light (1): on=1, off=0
red block (6): (x, y, z, euler_x, euler_y, euler_z)
blue block (6): (x, y, z, euler_x, euler_y, euler_z)
pink block (6): (x, y, z, euler_x, euler_y, euler_z)
```
The robot proprioceptive information, which also includes joint positions can be accessed with:
```
['robot_obs']
(dtype=np.float32, shape=(15,))
tcp position (3): x,y,z in world coordinates
tcp orientation (3): euler angles x,y,z in world coordinates
gripper opening width (1): in meter
arm_joint_states (7): in rad
gripper_action (1): binary (close = -1, open = 1)
```
### Language Annotations
The language annotations are in a subdirectory of the train and validation folders called `lang_annotations`.
The file `auto_lang_ann.npy` contains the language annotations and its embeddings besides of additional metadata such as the task id, the sequence indexes.
```
['language']['ann']: list of raw language
['language']['task']: list of task_id
['language']['emb']: precomputed miniLM language embedding
['info']['indx']: list of start and end indices corresponding to the precomputed language embeddings
```
The `embeddings.npy` file is only present on the validation folder, this file contains the embeddings used only during the Rollouts (test inference) to condition the policy.

## Visualize Language Annotations
We provide a script to generate a video that visualizes the language annotations of the recorded play data.
By default we visualize the first 100 sequences, but feel free to more sequences (just change this [line](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/utils/visualize_annotations.py#L57)).
A example video is.
```
cd $CALVIN_ROOT/calvin_models/calvin_agent
python utils/visualize_annotations.py datamodule.root_data_dir=$CALVIN_ROOT/dataset/task_D_D/ datamodule/observation_space=lang_rgb_static
```
