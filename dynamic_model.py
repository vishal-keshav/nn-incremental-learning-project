"""Define model architecture here.
The model class must be a torch.nn.Module and should have forward method.
For custome model, they are needed to be treated accordingly in train function.
"""

import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

"""
Some terminology:

Each task is composed of a series of modules.
If there are K modules in a task, then input layer (where input is a tensor)
is indexed 0 and the the last layer that produces the output is indexed K-1.
Each of this indexes are called levels. Eg. if level=2, we are talking about
third module from the input module.

Each module is composed of a series of blocks. The information of the block
is provided in a list called block_list. Each element of a block_list is a 
tuple (called block_info). This tuple looks like the following:

(
    'conv',
    param1,
    param2,
    ..
)
based on the layer name (first entry), the tuple has to contain the parameters.
"""


class ModuleFactory(object):
    def __init__(self, layer_list_info):
        """
        Initialization of the the factory
        layer_list_info is a dictionary with layer level as key
        and a list describing all layer information as value.
        Currently, no checks are made if the layer information is valid.
    
        Arguments:
            layer_list_info: dict
        Returns:
            None
        """
        self.layer_list_info = layer_list_info
    
    def _construct_block(self, block_info):
        """
        Constructs a neural network layer based on information in block_info
        Only three operations are application as of now!

        Arguments:
            block_info: tuple
        Returns:
            block: nn.Module
        """
        layer_name = block_info[0]
        if layer_name=='Conv2d':
            in_channels, out_channels, kernel_size = block_info[1:]
            return nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size)
        elif layer_name=='ReLU':
            return nn.ReLU(inplace=True)
        elif layer_name=='MaxPool2d':
            kernel_size, stride = block_info[1:]
            return nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride)
        elif layer_name=='BatchNorm2d':
            num_features = block_info[1]
            return nn.BatchNorm2d(num_features=num_features)
        elif layer_name=='Linear':
            in_features, out_features = block_info[1:]
            return nn.Linear(in_features=in_features,
                          out_features=out_features)
        else:
            raise Exception("_construct_block cannot construct block")

    def generate_module(self, level):
        """
        Constructs a module based on the info indexed at level idx in 
        layer_list_info dictionary

        Arguments:
            level: int
        Returns:
            final_module: nn.Module
            nr_params_in_module: int
        """
        layer_info = self.layer_list_info[level]
        block_list = []
        for block_info in layer_info:
            block_list.append(self._construct_block(block_info))
        final_module = nn.ModuleList(block_list)
        nr_params_in_module = sum(p.numel() for p in final_module.parameters())
        return final_module, nr_params_in_module
        

class DynaNet(nn.Module):   
    def __init__(self, layer_list_info):
        """
        The initialization does not add any parameter in the module.
        self.add_module() is done by add_model_for_task in the module dicts
        initialized here.

        Arguments:
            layer_list_info: dict
        Returns:
            None
        """
        super(DynaNet, self).__init__()
        self.layer_list_info = layer_list_info
        self.task_modules = nn.ModuleDict()
        self.classification_layers = nn.ModuleDict()
        self.module_generator = ModuleFactory(layer_list_info)
        self.task_module_name_path = {}
        self.nr_levels = len(layer_list_info)
        self.task_idx = None
    
    def add_model_for_task(self, task_idx):
        """
        Adds the modules for a specific task.
        It is assumed that this is called with task_idx = 0, 1, 2,... in this order
        If the task_idx = 0, all modules are added.
        else only one module is added (right now it is the last module)
        Classification layer is added for each task.
        Returns number of new parameters added.
        
        Arguments:
            task_idx: int
        Returns:
            total_params: int
        """
        level_path = []
        total_params = 0
        nr_params = 0
        if task_idx==0:
            for level in range(self.nr_levels):
                str_handle = "{}_{}".format(task_idx, level)
                level_path.append(str_handle)
                self.task_modules[str_handle], nr_params = self.module_generator.generate_module(level)
                total_params += nr_params
        else:
            # For now, just add a module at the end
            for level in range(self.nr_levels-1):
                str_handle = "{}_{}".format(0, level)
                level_path.append(str_handle)
            str_handle = "{}_{}".format(task_idx, self.nr_levels-1)
            level_path.append(str_handle)
            self.task_modules[str_handle], nr_params = self.module_generator.generate_module(self.nr_levels-1)
            total_params += nr_params
        self.task_module_name_path[task_idx] = level_path
        self.classification_layers[str(task_idx)] = nn.Linear(144, 10)
        self._set_task(task_idx)
        # print("{} parameters added".format(total_params))
        return total_params
    
    def _set_task(self, task_idx):
        """
        User of this function is add_model_for_task.
        It sets the task path that will be taken when forward is called.

        For testing, it can be set from the user of this class object.

        Arguments:
            task_idx: int
        Returns:
            None
        """
        self.task_idx = task_idx

    # Defining the forward pass
    def forward(self, x):
        """
        Defines the forward pass for the selected task_idx
        """
        for task_module_name in self.task_module_name_path[self.task_idx]:
            for layer in self.task_modules[task_module_name]:
                x = layer(x)
            #x = self.task_modules[task_module_name](x)
        x = x.view(x.size(0), -1)
        x = self.classification_layers[str(self.task_idx)](x)
        return x

def param_per_task_group_helper(task_id, model):
    """
    A helper function that is going to group parameters used during training
    a particular task. Optimizier will use this helper function.

    The implementation is ad-hoc, meaning it is based on the module names we
    know is going to be constructed by the DynaNet.

    Arguments:
        task_id: int
        model: nn.Module
    Returns:
        parameters: List[nn.Parameters]
    """
    # The first task is special case
    param_list = []
    nr_layers = model.nr_levels
    if task_id==0:
        for i in range(nr_layers):
            module_name = '{}_{}'.format(0, i)
            param_list.extend(list(model._modules['task_modules'][module_name].parameters()))
    else:
        module_name = '{}_{}'.format(task_id, nr_layers-1)
        param_list.extend(list(model._modules['task_modules'][module_name].parameters()))
    param_list.extend(list(model._modules['classification_layers'][str(task_id)].parameters()))
    return param_list


#################################### Test ######################################

def test_module_factory():
    layer_list_info = {
        0: [('Conv2d', 3, 16, 3), ('BatchNorm2d', 16), ('ReLU',), ('MaxPool2d', 2, 2)],
        1: [('Conv2d', 16, 4, 3), ('BatchNorm2d', 4), ('ReLU',), ('MaxPool2d', 2, 2)]
        #2: [('Linear', 256, 10)]
    }
    model_factory_obj = ModuleFactory(layer_list_info)
    layer1 = model_factory_obj.generate_module(0)
    print(layer1)
    layer2 = model_factory_obj.generate_module(1)
    print(layer2)
    layer3 = model_factory_obj.generate_module(2)
    print(layer3)

def test_DynaNet():
    layer_list_info = {
        0: [('Conv2d', 3, 16, 3), ('BatchNorm2d', 16), ('ReLU',), ('MaxPool2d', 2, 2)],
        1: [('Conv2d', 16, 4, 3), ('BatchNorm2d', 4), ('ReLU',), ('MaxPool2d', 2, 2)]
        #2: [('Linear', 256, 10)]
    }
    net = DynaNet(layer_list_info)
    print(net)
    net.add_model_for_task(0)
    print(net)
    x = torch.ones(1,3,32,32)
    y = net(x)
    net.add_model_for_task(1)
    print(net)
    x = torch.ones(1,3,32,32)
    y = net(x)
    net.add_model_for_task(2)
    print(net)
    x = torch.ones(1,3,32,32)
    y = net(x)

def test_param_group_helper():
    layer_list_info = {
        0: [('Conv2d', 3, 16, 3), ('BatchNorm2d', 16), ('ReLU',), ('MaxPool2d', 2, 2)],
        1: [('Conv2d', 16, 4, 3), ('BatchNorm2d', 4), ('ReLU',), ('MaxPool2d', 2, 2)]
        #2: [('Linear', 256, 10)]
    }
    net = DynaNet(layer_list_info)
    print(net)
    net.add_model_for_task(0)
    net.add_model_for_task(1)
    net.add_model_for_task(2)
    print(net)
    #for m in net.named_modules():
    #    print("M", m)
    param_t0 = param_per_task_group_helper(0, net)
    print("For task 0")
    for p in param_t0:
        print(type(p))
    param_t1 = param_per_task_group_helper(1, net)
    print("For task 1")
    for p in param_t1:
        print(type(p))
    param_t2 = param_per_task_group_helper(2, net)
    print("For task 2")
    for p in param_t2:
        print(type(p))




if __name__ == "__main__":
    #test_module_factory()
    #test_DynaNet()
    test_param_group_helper()