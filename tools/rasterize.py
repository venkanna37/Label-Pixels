import ogr
import osr
import gdal
import numpy as np
import os, sys
import glob
import argparse


def rasterize(args):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    raster_layer = gdal.Open(args.raster)
    vector_layer = driver.Open(args.vector, 0)

    # (1)Check Projection of vector and raster datasets
    ds = vector_layer.GetLayer().GetSpatialRef()
    vector_epsg = ds.GetAttrValue('AUTHORITY', 1)
    proj = osr.SpatialReference(wkt=raster_layer.GetProjection())
    raster_epsg = proj.GetAttrValue('AUTHORITY', 1)
    if vector_epsg != raster_epsg:
        print("The projections of Vector and Raster files are not same")
        sys.exit()

    # (2) Creating the destination raster data source
    output_file = args.output_file
    ref_image = gdal.Open(args.raster)
    gt = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()
    [cols, rows] = np.array(ref_image.GetRasterBand(1).ReadAsArray()).shape
    target_ds = gdal.GetDriverByName('GTiff').Create(output_file, rows, cols, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gt)
    target_ds.SetProjection(proj)
    target_ds.GetRasterBand(1)

    v_layer = vector_layer.GetLayer()
    featureTest = v_layer.GetNextFeature()
    geom = featureTest.GetGeometryRef()
    # print(geom.GetGeometryType(), ogr.wkbLineString)

    # (3) Creating polygon feature from line feature with buffer
    if geom.GetGeometryType() == ogr.wkbLineString:

        # Buffering
        file = '../data/buffer.shp'
        if os.path.exists(file):
            driver.DeleteDataSource(file)
        ds = driver.CreateDataSource(file)
        if ds is None:
            print('Could not create file')
            sys.exit(1)

        geosr = osr.SpatialReference()
        geosr.ImportFromEPSG(int(vector_epsg))
        lyrBuffer = ds.CreateLayer('buffer', geom_type=ogr.wkbPolygon, srs=geosr)
        featureDefn = lyrBuffer.GetLayerDefn()

        while featureTest:
            geomTest = featureTest.GetGeometryRef()
            # geomTest.Transform(geosr2utmsr)
            if args.buffer_atr:
                try:
                    buffer_width = (float(featureTest.GetField(args.buffer_atr)) * args.buffer)
                except ValueError:
                    print("The type of attribute value should be integer or float")
                    sys.exit()
                except KeyError:
                    print("Attribute does not exist in shapefile attributes")
                    sys.exit()
            else:
                buffer_width = args.buffer
            geomBuffer = geomTest.Buffer(buffer_width)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(geomBuffer)
            lyrBuffer.CreateFeature(outFeature)
            outFeature.Destroy()
            featureTest = v_layer.GetNextFeature()
        gdal.RasterizeLayer(target_ds, [1], lyrBuffer)
        target_ds = None
        ds = None

    elif geom.GetGeometryType() == ogr.wkbPolygon:
        lyrBuffer = v_layer
        if args.label_atr:
            label_atr = ["ATTRIBUTE=" + args.label_atr]
            # gdal.RasterizeLayer(ds,bands,layer,burn_values, options = ["BURN_VALUE_FROM=Z"])
            gdal.RasterizeLayer(target_ds, [1], lyrBuffer, options=label_atr)
            target_ds = None
        else:
            gdal.RasterizeLayer(target_ds, [1], lyrBuffer)
            target_ds = None

    else:
        print("The Geometry type is neither Polygon nor Line")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", type=str, help="Raster file name  with directory")
    parser.add_argument("--vector", type=str, help="Vector file name  with directory ")
    parser.add_argument("--buffer", type=int, help="Buffer width of line feature", default=1)
    parser.add_argument("--output_file", type=str, help="Output image name with directory")
    parser.add_argument("--buffer_atr", type=str, help="Attribute from the vector file, This attribute can be"
                                                       " buffer Width or It multiplies with buffer", default=None)
    parser.add_argument("--label_atr", type=str, help="Attribute from the vector file to assign label to pixel."
                                                      "Not required for single class")
    args = parser.parse_args()
    rasterize(args)
