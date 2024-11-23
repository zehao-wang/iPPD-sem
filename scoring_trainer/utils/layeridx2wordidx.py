# roomidx2name = {
#     17: 'meetingroom', 
#     19: 'office', 
#     12: 'hallway',
#     15: 'kitchen', 
#     29: 'room', 
#     1:  'classroom', 
#     18: 'lounge', 
#     13: 'library', 
#     2:  'dining booth', 
#     21: 'rec/game', 
#     3:  'spa/sauna', 
#     22: 'stairs', 
#     5:  'bathroom', 
#     8:  'dining room', 
#     10: 'familyroom', 
#     16: 'living room', 
#     9:  'lobby', 
#     6:  'bedroom', 
#     14: 'laundryroom', 
#     7:  'closet', 
#     23: 'toilet', 
#     20: 'terrace', 
#     28: 'balcony', 
#     24: 'toolroom', 
#     4:  'junk', 
#     26: 'gym', 
#     25: 'tv', 
#     11: 'garage', 
#     0:  'bar', 
#     27: 'outdoor'
# }

# room_word_mapping = {
#     'meetingroom': 1474,  # office
#     'office': 1474,  # office
#     'hallway': 1036, # hallway
#     'kitchen': 1205, # kitchen 
#     'room': 1842, # room 
#     'classroom': 469, # classroom 
#     'lounge': 1311, # lounge 
#     'library': 1265, # library
#     'dining booth': 660, # dining 
#     'rec/game': 1842,  # room
#     'spa/sauna': 2029, # spa 
#     'stairs': 2058,  # stairs
#     'bathroom': 222, # bathroom 
#     'dining room': 660, # dining 
#     'familyroom': 1842, # room
#     'living room': 1291, # livingroom
#     'lobby': 1294, # lobby 
#     'bedroom': 246, # bedroom
#     'laundryroom': 1235,  # laundry
#     'closet': 482, # closet
#     'toilet': 2248,  # toilet
#     'terrace': 2196, # terrace
#     'balcony': 183, # balcony 
#     'toolroom': 1842, # room
#     'junk': 2267,  # trash
#     'gym': 1027,  # gym
#     'tv': 2306, 
#     'garage': 955, 
#     'bar': 197, 
#     'outdoor': 1524
# }

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
    "floor": 883,
    "ceiling": 414,
    "objects": 1469,
    "misc": 1454, # "no"
    "chair": 424, # "chair"
    "door": 701, # "door"
    "table": 2159, # table
    "picture": 1634, # picture
    "cabinet": 375, # cabinet
    "cushion": 606, # cushhion
    "window": 2449, # window
    "sofa": 2020, # sofa
    "bed": 243, # sofa
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

obj_layeridx2wordidx = {
   k: obj_word_mapping[v] for k,v in semantic_sensor_40cat.items() if v in obj_word_mapping
}

import numpy as np
print([semantic_sensor_40cat[i] for i in range(40) if semantic_sensor_40cat[i] not in obj_word_mapping ])
gt_obj_id2word_idx = np.array([obj_word_mapping[semantic_sensor_40cat[i]] for i in range(40)])

seg_obj_id2word_idx = np.array([778, 181, 243, 375, 414, 424, 476, 553, 598, 701, 849, 883, 1198, 1226,
    1266, 1383, 1390, 1198, 1532, 1644, 1667, 1799, 1940, 1972, 2020, 2058, 2159, 2247, 2248,
    2261, 2306, 2348, 2392, 2449])