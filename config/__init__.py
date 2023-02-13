from .defaults import _C as cfg

TYPE_ID_CONVERSION = {
    'CAR': 0,
    'PD': 1,
    'Rider': 2,
    'BUS': 3,
    'TRUCK': 4,
    'Three': 5,
    'trailerback': 6,
    'VAN':7,
}
TYPE_ID_INVERSE = {
    0:'CAR',
    1:'PD',
    2:'Rider',
    3:'BUS',
    4:'TRUCK',
    5:'Three',
    6:'trailerback',
    7:'VAN',
}

TYPE_ID_REG_WEIGHT = {
    'CAR': 1.0,
    'PD': 1.0,
    'Rider': 1.0,
    'BUS': 1.0,
    'TRUCK': 1.0,
    'Three': 1.0,
    'trailerback': 1.0,
    'VAN':1.0,
}

TYPE_ID_COLOR = {
    "VAN" : (0, 0, 255),
    "CAR": (0, 255, 0),
    "PD": (255, 0, 0),
    "Rider": (0, 122, 255),
    "Three": (0, 122, 122),
    "TRUCK":(122, 122, 255),
    "BUS": (122, 122, 0),
    "SPECIALCAR":(122, 0, 122),
    "trailerback":(122, 0, 0),
    "VAN":(255, 122, 0),
    
}


# TYPE_ID_CONVERSION = {
#     'Car': 0,
#     'Pedestrian': 1,
#     'Cyclist': 2,
#     'Van': -4,
#     'Truck': -4,
#     'Person_sitting': -2,
#     'Tram': -99,
#     'Misc': -99,
#     'DontCare': -1,
# }