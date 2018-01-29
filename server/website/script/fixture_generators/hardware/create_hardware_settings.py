#
# OtterTune - create_hardware_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import csv
import json
import shutil
from collections import OrderedDict


HW = OrderedDict()
with open('ec2_instance_types.csv', 'r') as f:
    READER = csv.reader(f)
    for i, row in enumerate(READER):
        if i == 0:
            header = row
        else:
            entry = {}
            entry['type'] = i + 1
            entry['name'] = row[0]
            entry['cpu'] = int(row[1])
            entry['memory'] = float(row[2].replace(',', ''))
            storage_str = row[3]
            storage_type = None
            if 'EBS' in storage_str:
                storage_type = 'EBS'
            elif 'NVMe' in storage_str:
                storage_type = 'NVMe'
            elif 'SSD' in storage_str:
                storage_type = 'SSD'
            elif entry['name'].startswith('r4'):
                storage_type = 'EBS'
            elif entry['name'].startswith('d2'):
                storage_type = 'HDD'
            elif entry['name'] == 'f1.16xlarge':
                storage_type = 'SSD'
            else:
                raise Exception('Unknown storage type for {}'.format(entry['name']))
            storage_list = None
            if storage_type == 'EBS':
                entry['storage'] = '40,40'
            elif entry['name'] == 'f1.2xlarge':
                entry['storage'] = storage_str.split(' ')[0]
            else:
                parts = storage_str.split(' ')
                num_devices = 4 if int(parts[0]) > 4 else int(parts[0])
                size = parts[2].replace(',', '')
                entry['storage'] = ','.join([size for _ in range(num_devices)])

            entry['storage_type'] = storage_type
            entry['additional_specs'] = json.dumps(OrderedDict(
                zip(header[4:], row[4:])), encoding='utf-8')
            HW[entry['name']] = entry

# For types.HardwareTypes
HW_CONSTS = [('GENERIC', 1, 'generic')]
for k, v in HW.iteritems():
    HW_CONSTS.append(('EC2_{}'.format(k.replace('.', '').upper()), v['type'], k))
HW_STR = ' '.join(['{} = {};'.format(k, v) for k, v, _ in HW_CONSTS])
TYPE_NAMES = ', '.join(['{}: \'{}\''.format(k, n) for k, _, n in HW_CONSTS])
with open('hardware_types.txt', 'w') as f:
    f.write(HW_STR + '\n')
    f.write('TYPE_NAMES = {' + TYPE_NAMES + '}')

ENTRIES = []
for k, v in HW.iteritems():
    ENTRIES.append({
        "model": "website.Hardware",
        'fields': v
    })

with open('hardware.json', 'w') as f:
    json.dump(ENTRIES, f, encoding='utf-8', indent=4)

shutil.copy('hardware.json', '../../preload/hardware.json')
