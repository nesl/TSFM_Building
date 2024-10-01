
building_condition = [
    ('off', 'unoccupied'), ('heat', 'unoccupied'), ('heat', 'occupied')
]

building_dura_pred_sr_tuple = [
    (24, 4, 300), (36, 4, 300), (48, 4, 300), (96, 4, 300),
    (24, 6, 600), (36, 6, 600), (48, 6, 600), (96, 6, 600),
    (24, 12, 900), (36, 12, 900), (48, 12, 900), (96, 12, 900),
    (24, 24, 1800), (36, 24, 1800), (48, 24, 1800), (96, 24, 1800)
]

electricity_condition = [
    ('off', 'unoccupied'), ('off', 'occupied')
]

electricity_dura_pred_sr_tuple = [
    (24, 4, 300), (36, 4, 300), (48, 4, 300), (96, 4, 300),
    (24, 6, 600), (36, 6, 600), (48, 6, 600), (96, 6, 600),
    (24, 12, 900), (36, 12, 900), (48, 12, 900), (96, 12, 900),
    (24, 24, 1800), (36, 24, 1800), (48, 24, 1800), (96, 24, 1800)
]

house_ids_uci = [20, 54, 88, 101, 102, 106, 116, 125, 127, 132, 152, 164, 181, 188, 193, 208, 209, 212, 221, 224, 239, 252, 253, 259, 272, 306, 310, 315, 337, 357]

electricity_uci_condition = [
    (None, id) for id in house_ids_uci
] # test for house id from house_ids

# electricity_uci_dura_pred_sr_tuple = [
#     (48, 12, 900), (72, 12, 900), (96, 12, 900),
#     (48, 24, 900), (72, 24, 900), (96, 24, 1800) ,
#     (168, 24, 1800),  
# ]
electricity_uci_dura_pred_sr_tuple = [
    (168, 24, 3600),  
]

house_ids_pecan = [i for i in range(1, 26)]

pecan_condition = [
    (None, id) for id in house_ids_pecan
] # test for house id from house_ids

pecan_dura_pred_sr_tuple = [
    (48, 12, 900), (72, 12, 900), (96, 12, 900),
    (48, 24, 900), (72, 24, 900), (96, 24, 1800) ,
    (168, 24, 1800),  
]

house_ids_umass = [33, 20, 103, 70, 106, 81, 75, 104, 110, 14, 18, 72, 8, 111, 83, 71, 30, 55, 40, 77, 94, 35, 54, 58, 91, 10, 63, 68, 86, 50]
umass_condition = [
    (None, id) for id in house_ids_umass
] # test for house id from house_ids
# umass_dura_pred_sr_tuple = [
#     (48, 12, 900), (72, 12, 900), (96, 12, 900),
#     (48, 24, 900), (72, 24, 900), (96, 24, 1800) ,
#     (168, 24, 1800),  
# ]
umass_dura_pred_sr_tuple = [
    (168, 24, 3600),  
]

house_ids_elecdemand = [1]
elecdemand_condition = [
    (None, id) for id in house_ids_elecdemand
]
elecdemand_dura_pred_sr_tuple = [
    (168, 24, 1800),  
]

house_ids_subseasonal = [1]
subseasonal_condition = [
    (None, id) for id in house_ids_elecdemand
]
subseasonal_dura_pred_sr_tuple = [
    (168*24, 24*24, 3600*24),  # in [hrs, hrs, seconds]
]

house_ids_loop_seattle = [1]
loop_seattle_condition = [
    (None, id) for id in house_ids_elecdemand
]
loop_seattle_dura_pred_sr_tuple = [
    (14, 2, 5*60),  # in [hrs, hrs, seconds]
    # (72, 6, 5*60)
]

house_ids_pems04 = [1]
pems04_condition = [
    (None, id) for id in house_ids_elecdemand
]
pems04_dura_pred_sr_tuple = [
    (14, 2, 5*60),  # in [hrs, hrs, seconds]
]

house_ids_rlp = [1]
rlp_condition = [
    (None, id) for id in house_ids_elecdemand
]
rlp_dura_pred_sr_tuple = [
    (84, 12, 30*60),  # in [hrs, hrs, seconds]
]

house_ids_covid = [1]
covid_condition = [
    (None, id) for id in house_ids_elecdemand
]
covid_dura_pred_sr_tuple = [
    (158*24, 24*24, 3600*24),  # in [hrs, hrs, seconds]
]

house_ids_c2000 = [1]
c2000_condition = [
    (None, id) for id in house_ids_elecdemand
]
c2000_dura_pred_sr_tuple = [
    (1008, 144, 3600*6),  # in [hrs, hrs, seconds]
]

restaurant_condition = [
    (None, id) for id in [1]
]
restaurant_dura_pred_sr_tuple = [
    (168*24, 24*24, 3600*24),  # in [hrs, hrs, seconds]
]

air_condition = [
    (None, id) for id in [1]
]
air_dura_pred_sr_tuple = [
    (168, 24, 3600),  # in [hrs, hrs, seconds]
]


#range house_ids from 1 to 33.
house_ids = [i for i in range(1, 33)]

ecobee_condition = [
    (None, id) for id in house_ids
] # test for house id from house_ids

# ecobee_dura_pred_sr_tuple = [
#     (24, 4, 300),  (36, 4, 300), 
#     (36, 6, 600), (48, 6, 600),
#     (48, 12, 900), (96, 12, 900), 
#     (168, 24, 1800),
# ]
ecobee_dura_pred_sr_tuple = [
    (168, 24, 3600),
]
elec_uci_indices = [138311, 42003, 42131, 44387, 44763, 46313, 47941, 51132, 51407, 55279, 57543, 58119, 58342, 58886, 59265, 59635, 62110, 65534, 66454, 67582, 69963, 72519, 73083, 74487, 76276, 78570, 78609, 82343, 83601, 85161, 87483, 87967, 88885, 91301, 93214, 93342, 93799, 96575, 97196, 99149, 101675, 103352, 104209, 104896, 105539, 109346, 112172, 114735, 114957, 117145, 117742, 118844, 119290, 121896, 124875, 125786, 125945, 128004, 129947, 130249, 130525, 135699, 136917, 136966, 138311]
elec_uci_season = ['Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Fall', 'Fall', 'Fall', 'Winter', 'Winter', 'Winter', 'Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Fall', 'Fall', 'Winter', 'Winter', 'Winter', 'Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Fall', 'Fall', 'Fall', 'Winter']