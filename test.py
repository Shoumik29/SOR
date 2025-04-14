from Backbones import backbone

model = backbone.get_backbone(name = 'vgg16')

model.summary()