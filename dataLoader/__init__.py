from .blender import BlenderDataset

from .tensoIR_rotation_setting import TensoIR_Dataset_unknown_rotated_lights
from .tensoIR_relighting_test import tensoIR_Relighting_test
from .tensoIR_simple import TensoIR_Dataset_simple
from .tensoIR_material_editing_test import tensoIR_Material_Editing_test
from .tensoIR_general_multi_lights import TensoIR_Dataset_unknown_general_multi_lights


dataset_dict = {'blender': BlenderDataset,
                'tensoIR_unknown_rotated_lights':TensoIR_Dataset_unknown_rotated_lights,
                'tensoIR_unknown_general_multi_lights': TensoIR_Dataset_unknown_general_multi_lights,
                'tensoIR_relighting_test':tensoIR_Relighting_test,
                'tensoIR_material_editing_test':tensoIR_Material_Editing_test,
                'tensoIR_simple':TensoIR_Dataset_simple,
                }
