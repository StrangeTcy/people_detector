import os
import requests





def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    if not os.path.exists(path):
        print ("You don't seem to have the weights at {}. \
                We'll download them now to {}. This should only happen once".
                format(path, path))
        r = requests.get('https://pjreddie.com/media/files/yolov3.weights', allow_redirects=True)
        open(path, 'wb').write(r.content)

    else:
        file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

