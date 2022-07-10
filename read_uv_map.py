import json

uv_map_coords = []
with open("uv_map.json") as json_file:
    data = json.load(json_file)
    for i in range(0,468):
        uv_map_coords.append((float(data["u"][str(i)]), float(data["v"][str(i)])))