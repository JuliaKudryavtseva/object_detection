from torchvision.models import ResNet18_Weights
from torchvision        import models

from torchview import draw_graph

import torch.nn.init as init
import torch.nn      as nn
import numpy         as np

import torch
import copy


class SSD_ResNet18(nn.Module):
    def __init__(self, num_bboxes_s, num_labels=3):
        super().__init__()

        self.num_labels = num_labels
        self.num_bboxes_s = num_bboxes_s

        self.used_layer_id_s = list(range(7, 13))
        self.num_bboxes_s = [6, 6, 6, 6, 6, 6]
                
        base_layers       = self._build_base_layers ()
        extra_layers      = self._build_extra_layers()
        self.total_layers = base_layers + extra_layers

        self.conf_layers, self.loc_layers = self._build_conf_loc_layers()
        
    def _build_base_layers(self):
        backbone_model    = models.resnet18(weights=ResNet18_Weights.DEFAULT)  #False

        drop_layers = ['layer4', 'avgpool', 'fc']
        base_layers = [layer for name, layer in backbone_model.named_children() if name not in drop_layers]

        conv256 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(conv256.weight)
        init.zeros_         (conv256  .bias)


        basic_layer = copy.deepcopy(getattr(backbone_model, 'layer2'))
        basic_layer[0].conv1 = conv256
        basic_layer[0].downsample[0] = copy.deepcopy(conv256)
        base_layers.append(basic_layer)

        return nn.ModuleList(base_layers)

    def _build_extra_layers(self, number_blocks=5):
        extra_layers = []

        conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1            )
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1 )
        relu2 = nn.ReLU(inplace=True)
        
        init.xavier_uniform_(conv1 .weight)
        init.zeros_         (conv1 .bias  )
        init.xavier_uniform_(conv2 .weight)
        init.zeros_         (conv2 .bias  )
        
        extra_layers = nn.Sequential(conv1, relu1, conv2, relu2)

        return [extra_layers for _ in range(number_blocks)]
    
    def _build_conf_loc_layers(self):
        
        conf_layers, loc_layers = [], []
        for i, j in enumerate(self.used_layer_id_s):

            if j > 7:
                _out_channels = self.total_layers[j][2].out_channels
            else:
                _out_channels =  self.total_layers[j][1].conv2.out_channels

            conf_layer = nn.Conv2d( _out_channels, self.num_bboxes_s[i] * self.num_labels, kernel_size=3, padding=1)
            loc_layer  = nn.Conv2d( _out_channels, self.num_bboxes_s[i] * 4              , kernel_size=3, padding=1)
            
            init.xavier_uniform_(conf_layer.weight)
            init.zeros_         (conf_layer  .bias)
            init.xavier_uniform_(loc_layer .weight)
            init.zeros_         (loc_layer   .bias)
            
            conf_layers += [conf_layer]
            loc_layers  += [loc_layer ]

        conf_layers = nn.ModuleList(conf_layers)
        loc_layers  = nn.ModuleList(loc_layers )
        
        return conf_layers, loc_layers
    
    def forward(self, x):
        source_s, loc_s, conf_s = [], [], []
        
        for i, current_layer in enumerate(self.total_layers):
            x = current_layer(x)
            if i > 6:
                source_s.append(x)
                
        for s, l, c in zip(source_s, self.loc_layers, self.conf_layers):
            conf_s.append(c(s).permute(0, 2, 3, 1).contiguous())
            loc_s .append(l(s).permute(0, 2, 3, 1).contiguous())
        conf_s = torch.cat([o.view(o.size(0), -1) for o in conf_s], 1)
        loc_s  = torch.cat([o.view(o.size(0), -1) for o in loc_s ], 1)
        
        conf_s = conf_s.view(conf_s.size(0), -1, self.num_labels)
        loc_s  = loc_s .view(loc_s .size(0), -1, 4              )

        return loc_s, conf_s


if __name__ == '__main__':
    num_bboxes_s = [6, 6, 6, 6, 6, 6]
    
    model = SSD_resnet18(num_bboxes_s, 3)
    model.eval()
     
    input_data  = torch.randn(1, 3, 720, 1280, dtype=torch.float, requires_grad=False)
    output_locs, output_confs = model( input_data )
    
    model_graph  = draw_graph(model, input_size=(1, 3, 720, 1280), expand_nested=True)
    visual_graph = model_graph.visual_graph
    graph_svg = visual_graph.pipe(format='png')
    with open('output_resnet18.png', 'wb') as f:
        f.write(graph_svg)

    print(output_locs .shape)
    print(output_confs.shape)
