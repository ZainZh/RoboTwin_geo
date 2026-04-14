from copy import deepcopy


# Task-bound default mappings from language placeholders to task actor attribute paths.
# These are used only when task_config.object_pointcloud.targets is omitted.
TASK_OBJECT_POINTCLOUD_TARGETS = {
    "hanging_mug": {
        "{A}": "mug",
        "{B}": "rack",
    },
    "beat_block_hammer": {
        "{A}": "hammer",
        "{B}": "block",
    },
    "pick_diverse_bottles": {
        "{A}": "bottle1",
        "{B}": "bottle2",
    },
}


def get_task_object_pointcloud_targets(task_name):
    if task_name is None:
        return None
    targets = TASK_OBJECT_POINTCLOUD_TARGETS.get(str(task_name))
    if targets is None:
        return None
    return deepcopy(targets)
