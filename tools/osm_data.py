"""
Script for downloading OpenStreetMap data to train deep learning models
the data can be downloaded using bbox coordinates and bbox coordinates for vector or raster files
Testing:
osmnx library has to install
Example command: python osm_data.py --bbox 30.33622 30.34384 78.03883 78.05268 --tags buildings
 --outfile ../data/dd_roads.geojson

The data has to clip with bounding box after downloading
"""
import osmnx as ox
import argparse


def download_osm(args):
    bbox_coord = tuple(args.bbox)
    filename = args.outfile
    # test  coordinates: north, south, east, west = 30.33622 ,30.34384,78.03883,78.05268
    north, south, east, west = bbox_coord[0], bbox_coord[1], bbox_coord[2], bbox_coord[3]
    print("Downloading: " + args.tags)
    if args.tags == "roads":
        tags = {'highway': True}
    elif args.tags == "buildings":
        tags = {'building': True}
    gdf = ox.geometries_from_bbox(north, south, east, west, tags)
    gdf = gdf.apply(lambda c: c.astype(str) if c.name != 'geometry' else c, axis=0)
    gdf.to_file(filename, driver='GeoJSON')
    print("Data downloaded and saved in " + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs='+', type=float, help="Bounding box coordinates, (north, south, east, west)")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4)
    parser.add_argument("--tags", type=str, help="Feature types ex: buildings, roads")
    parser.add_argument("--outfile", type=str, help="Output filename with directory")
    args = parser.parse_args()
    download_osm(args)
