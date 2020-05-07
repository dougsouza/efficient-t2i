import torch.nn as nn


def orthogonal(m):
    if isinstance(m, nn.Conv2d):
        # gain = nn.init.calculate_gain('relu')
        gain = nn.init.calculate_gain('linear')
        # if m.out_channels == 3:
        #     gain = nn.init.calculate_gain('linear')
        nn.init.orthogonal_(m.weight.data, gain=gain)
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('linear')
        nn.init.orthogonal_(m.weight.data, gain=gain)


def xavier_normal(m):
    if isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        if m.out_channels == 3:
            gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(m.weight.data, gain=gain)


def xavier_uniform(m):
    if isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        if m.out_channels == 3:
            gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(m.weight.data, gain=gain)
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(m.weight.data, gain=gain)


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
        if m.out_channels == 3:
            nn.init.kaiming_normal_(m.weight.data, a=1, nonlinearity='linear')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=1, nonlinearity='linear')


def kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, a=0, nonlinearity='relu')
        if m.out_channels == 3:
             nn.init.kaiming_uniform_(m.weight.data, a=1, nonlinearity='linear')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, a=1, nonlinearity='linear')


def normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)



__initdict__ = {'orthogonal': orthogonal,
                'xavier_uniform': xavier_uniform,
                'xavier_normal': xavier_normal,
                'kaiming_uniform': kaiming_uniform,
                'kaiming_normal': kaiming_normal,
                'normal': normal}



def get_initializer(name):
    return __initdict__[name]


def get_initializer_names():
    return __initdict__.keys()
