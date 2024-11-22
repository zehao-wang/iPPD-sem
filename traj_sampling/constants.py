from matplotlib import cm
import numpy as np
roomidx2name = {
    17: 'meetingroom', 
    19: 'office', 
    12: 'hallway',
    15: 'kitchen', 
    29: 'room', 
    1: 'classroom', 
    18: 'lounge', 
    13: 'library', 
    2: 'dining booth', 
    21: 'rec/game', 
    3: 'spa/sauna', 
    22: 'stairs', 
    5: 'bathroom', 
    8: 'dining room', 
    10: 'familyroom', 
    16: 'living room', 
    9: 'lobby', 
    6: 'bedroom', 
    14: 'laundryroom', 
    7: 'closet', 
    23: 'toilet', 
    20: 'terrace', 
    28: 'balcony', 
    24: 'toolroom', 
    4: 'junk', 
    26: 'gym', 
    25: 'tv', 
    11: 'garage', 
    0: 'bar', 
    27: 'outdoor'
}

semantic_sensor_40cat = {
    0: "wall",
    1: "floor",
    2: "chair",
    3: "door",
    4: "table",
    5: "picture",
    6: "cabinet",
    7: "cushion",
    8: "window",
    9: "sofa",
    10:"bed",
    11:"curtain",
    12:"drawer",
    13:"plant",
    14:"sink",
    15:"stairs",
    16:"ceiling",
    17:"toilet",
    18:"stool",
    19:"towel",
    20:"mirror",
    21:"tv_monitor",
    22:"shower",
    23:"column",
    24:"bathtub",
    25:"counter",
    26:"fireplace",
    27:"lighting",
    28:"beam",
    29:"railing",
    30:"shelving",
    31:"blinds",
    32:"gym_equipment",
    33:"seating",
    34:"board_panel",
    35:"furniture",
    36:"appliances",
    37:"clothes",
    38:"objects",
    39:"misc"
}

obj_merged_dict = {
    "wall": "wall",
    "floor": "floor",
    "chair": "seating",
    "door": "door",
    "table": "table",
    "picture": "picture",
    "cabinet": "closet", # 柜子
    "cushion": "seating",
    "window": "window",
    "sofa": "seating",
    "bed": "bed",
    "curtain": "curtain",
    "drawer": "closet",
    "plant": "plant",
    "sink": "sink",
    "toilet": "toilet",
    "stool": "seating",
    "towel": "towel",
    "mirror": "mirror",
    "tv_monitor": "tv_monitor",
    "shower": "shower",
    "column": "column",
    "bathtub": "bathtub",
    "counter": "counter",
    "fireplace": "fireplace",
    "lighting": "lighting",
    "beam": "lighting",
    "railing": "railing",
    "shelving": "counter",
    "blinds": "window",
    "gym_equipment": "gym_equipment",
    "seating": "seating",
    "board_panel": "board_panel",
    "furniture": "counter",
    "appliances": "appliances",
    "clothes": "clothes",
}


obj_word_mapping = {
    "wall": "wall",
    "floor": "floor",
    "chair": "seating",
    "door": "door",
    "table": "table",
    'desk': 'table',
    "picture": "picture",
    "cabinet": "closet", # 柜子
    "cushion": "seating",
    "window": "window",
    "sofa": "seating",
    "bed": "bed",
    "curtain": "curtain",
    "drawer": "closet",
    'shelves': 'closet',
    "plant": "plant",
    "flower": "plant",
    "sink": "sink",
    "toilet": "toilet",
    "stool": "seating",
    "towel": "towel",
    "mirror": "mirror",
    "tv monitor": "tv_monitor",
    "shower": "shower",
    "column": "column",
    "pillars": "column",
    "bathtub": "bathtub",
    "counter": "counter",
    "fireplace": "fireplace",
    "lighting": "lighting",
    "beam": "lighting",
    "railing": "railing",
    "shelving": "counter",
    "blinds": "window",
    "gym equipment": "gym_equipment",
    'treadmill': 'gym_equipment',
    "seating": "seating",
    "board panel": "board_panel",
    "furniture": "counter",
    "appliances": "appliances",
    "clothes": "clothes",
    'bannister': 'railing'
}

room_word_mapping = {
    'meetingroom': 'office', 
    'conferenceroom': 'office',
    'office': 'office', 
    'hallway': 'hallway',
    'doorway': 'hallway',
    'kitchen': 'kitchen', 
    'room': 'room', 
    'classroom': 'classroom', 
    'lounge': 'room', 
    'library': 'library', 
    'dining booth': 'dining room', 
    'rec': 'room',
    'game': 'room', 
    'spa':'room',
    'sauna': 'room', 
    'massage room': 'room',
    'stairs': 'stairs', 
    'bathroom': 'bathroom', 
    'dining room': 'dining room', 
    'familyroom': 'room', 
    'living room': 'living room', 
    'lobby': 'lobby', 
    'entryway': 'lobby',
    'foyer': 'lobby',
    'bedroom': 'bedroom', 
    'laundryroom': 'laundryroom', 
    'mudroom': 'laundryroom',
    'closet': 'closet', 
    'toilet': 'toilet',  # consider to remove
    'terrace': "balcony", 
    'porch': 'balcony',
    'deck': 'balcony',
    'balcony': 'balcony', 
    'toolroom': 'room', 
    'utilityroom': 'room',
    'junk': 'junk', 
    'gym': 'gym', 
    'workout': 'gym',
    'exercise': 'gym',
    'tv': 'room', 
    'garage': 'garage', 
    'bar': 'kitchen', 
    'outdoor': 'outdoor'  
}

room_merged_dict = {
    'meetingroom': 'office', 
    'office': 'office', 
    'hallway': 'hallway',
    'kitchen': 'kitchen', 
    'room': 'room', 
    'classroom': 'classroom', 
    'lounge': 'room', 
    'library': 'library', 
    'dining booth': 'dining room', 
    'rec/game': 'room', 
    'spa/sauna': 'room', 
    'stairs': 'stairs', 
    'bathroom': 'bathroom', 
    'dining room': 'dining room', 
    'familyroom': 'room', 
    'living room': 'living room', 
    'lobby': 'lobby', 
    'bedroom': 'bedroom', 
    'laundryroom': 'laundryroom', 
    'closet': 'closet', 
    'toilet': 'toilet',  # consider to remove
    'terrace': "balcony", 
    'balcony': 'balcony', 
    'toolroom': 'room', 
    'junk': 'junk', 
    'gym': 'gym', 
    'tv': 'room', 
    'garage': 'garage', 
    'bar': 'kitchen', 
    'outdoor': 'outdoor'
}

room_set = {name: i for i, name in enumerate(sorted(list(set(room_merged_dict.values()))))}
objs_set = set(obj_merged_dict.values())
objs_set.remove('closet')
objs_set = {name: i for i, name in enumerate(sorted(list(objs_set)))}


tab10_colors_rgb = (np.array(cm.get_cmap('tab10').colors)*255).astype(np.uint8)
tab10_colors_rgba =  np.concatenate([tab10_colors_rgb, np.ones((10,1), dtype=np.uint8)*255], axis=1)

roomidx2name = {
    17: 'meetingroom', 
    19: 'office', 
    12: 'hallway',
    15: 'kitchen', 
    29: 'room', 
    1:  'classroom', 
    18: 'lounge', 
    13: 'library', 
    2:  'dining booth', 
    21: 'rec/game', 
    3:  'spa/sauna', 
    22: 'stairs', 
    5:  'bathroom', 
    8:  'dining room', 
    10: 'familyroom', 
    16: 'living room', 
    9:  'lobby', 
    6:  'bedroom', 
    14: 'laundryroom', 
    7:  'closet', 
    23: 'toilet', 
    20: 'terrace', 
    28: 'balcony', 
    24: 'toolroom', 
    4:  'junk', 
    26: 'gym', 
    25: 'tv', 
    11: 'garage', 
    0:  'bar', 
    27: 'outdoor'
}

semantic_sensor_40cat = {
    0: "wall",
    1: "floor",
    2: "chair",
    3: "door",
    4: "table",
    5: "picture",
    6: "cabinet",
    7: "cushion",
    8: "window",
    9: "sofa",
    10:"bed",
    11:"curtain",
    12:"drawer",
    13:"plant",
    14:"sink",
    15:"stairs",
    16:"ceiling",
    17:"toilet",
    18:"stool",
    19:"towel",
    20:"mirror",
    21:"tv_monitor",
    22:"shower",
    23:"column",
    24:"bathtub",
    25:"counter",
    26:"fireplace",
    27:"lighting",
    28:"beam",
    29:"railing",
    30:"shelving",
    31:"blinds",
    32:"gym_equipment",
    33:"seating",
    34:"board_panel",
    35:"furniture",
    36:"appliances",
    37:"clothes",
    38:"objects",
    39:"misc"
}


obj_word_mapping = {
    "wall": 2392, # "wall"
    "chair": 424, # "chair"
    "door": 701, # "door"
    "table": 2159, # table
    "picture": 1634, # picture
    "cabinet": 375, # cabinet
    "cushion": 606, # cushhion
    "window": 2449, # window
    "sofa": 2020, # sofa
    "bed": 243, # bed
    "curtain": 598, # curtain
    "drawer": 728, # drawer
    "plant": 1667, # plant
    "sink": 1972, # sink
    "stairs": 2058, # stairs
    "toilet": 2248, # toilet
    "stool": 424, # chair
    "towel": 2261, # towel
    "mirror": 1390, # mirror
    "tv_monitor": 2306, # tv
    "shower": 1951, # shower
    "column": 501, # column
    "bathtub": 224, # bathtub
    "counter": 553, # counter
    "fireplace": 867, # fireplace
    "lighting": 1270, # lighting
    "beam": 231, # beam
    "railing": 1766, # railing
    "shelving": 1941, # shelving
    "blinds": 290, # blinds
    "gym_equipment": 2274, # treadmill
    "seating": 424, # chair
    "board_panel": 1561, # panel
    "furniture": 945, # furniture
    "appliances": 119, # appliances
    "clothes":486, # clothes
}

room_word_mapping = {
    'meetingroom': 1474,  # office
    'office': 1474,  # office
    'hallway': 1036, # hallway
    'kitchen': 1205, # kitchen 
    'room': 1842, # room 
    'classroom': 469, # classroom 
    'lounge': 1311, # lounge 
    'library': 1265, # library
    'dining booth': 660, # dining 
    'rec/game': 1842,  # room
    'spa/sauna': 2029, # spa 
    'stairs': 2058,  # stairs
    'bathroom': 222, # bathroom 
    'dining room': 660, # dining 
    'familyroom': 1842, # room
    'living room': 1291, # livingroom
    'lobby': 1294, # lobby 
    'bedroom': 246, # bedroom
    'laundryroom': 1235,  # laundry
    'closet': 482, # closet
    'toilet': 2248,  # toilet
    'terrace': 2196, # terrace
    'balcony': 183, # balcony 
    'toolroom': 1842, # room
    'junk': 2267,  # trash
    'gym': 1027,  # gym
    'tv': 2306, 
    'garage': 955, 
    'bar': 197, 
    'outdoor': 1524
}

# obj = {'chair', 'door', 'table', 'picture', 'cabinet', 'cushhion', 'window', 'sofa', 'bed', 'curtain', 'drawer', 'plant', 'sink', 'stairs', 'toilet', 'towel' ,'mirror', 'tv', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam', 'railing', 'shelving', 'blinds', 'treadmill', 'panel', 'furniture', 'appliances' , 'clothes'}
# rooms = {'office', 'hallway', 'kitchen', 'room', 'classroom', 'lounge', 'library', 'dining', 'spa', 'stairs', 'bathroom', 'livingroom', 'lobby', 'bedroom', 'laundry', 'closet', 'toilet', 'terrace', 'balcony', 'trash', 'gym', 'garage', 'outdoor', 'bar'}

obj_layeridx2wordidx = {
   k: obj_word_mapping[v] for k,v in semantic_sensor_40cat.items() if v in obj_word_mapping
}

obj_layeridx2wordidx.update({-1: 597}) # for notify agent use word 'current'

room_layeridx2wordidx = {
   k: room_word_mapping[v] for k,v in roomidx2name.items() if v in room_word_mapping
}