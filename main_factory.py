from sdeep.cli import sdeepModules


args = {
    'dncnn_layers': 8,
    'dncnn_channels': 1,
    'dncnn_features': 64
}
DnCNN = sdeepModules.get_instance('DnCNN', args)
print(DnCNN)
