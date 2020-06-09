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

    # Projections to add buffer in meters
    # geosr = osr.SpatialReference()
    # geosr.ImportFromEPSG(4326)
    # utmsr = osr.SpatialReference()
    # utmsr.ImportFromEPSG(32643)
    # geosr2utmsr = osr.CoordinateTransformation(geosr, utmsr)
    # utmsr2geosr = osr.CoordinateTransformation(utmsr, geosr)
    # base = os.path.basename(vector)
    # file_name, ext = os.path.splitext(base)
    # print('Rasterizing:  ' + file_name)
    # test = ogr.Open(vector, 0)
    # lyrTest = test.GetLayer()
    # v_proj = lyrTest.GetSpatialRef()

    # check units
    ds = vector_layer.GetLayer().GetSpatialRef()
    vector_epsg = ds.GetAttrValue('AUTHORITY', 1)
    # print(vector_epsg, type(vector_epsg))
    proj = osr.SpatialReference(wkt=raster_layer.GetProjection())
    raster_epsg = proj.GetAttrValue('AUTHORITY', 1)
    if vector_epsg != raster_epsg:
        print("The projection of Vector and Raster files is not same")
        sys.exit()

    # Buffering
    file = '..\\data\\buffer.shp'
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

    v_layer = vector_layer.GetLayer()
    featureTest = v_layer.GetNextFeature()
    while featureTest:
        geomTest = featureTest.GetGeometryRef()
        # geomTest.Transform(geosr2utmsr)
        if args.attribute:
            buffer_width = (float(featureTest.GetField(args.attribute)) * args.buffer)
        else:
            buffer_width = args.buffer
        geomBuffer = geomTest.Buffer(buffer_width)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        lyrBuffer.CreateFeature(outFeature)
        outFeature.Destroy()
        featureTest = v_layer.GetNextFeature()

    # Rasterize
    output_file = args.output_file
    ref_image = gdal.Open(args.raster)
    gt = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()

    # 2) Creating the destination raster data source
    # pixelWidth = pixelHeight = gt[1]  # depending how fine you want your raster ##COMMENT 1
    [cols, rows] = np.array(ref_image.GetRasterBand(1).ReadAsArray()).shape
    target_ds = gdal.GetDriverByName('GTiff').Create(output_file, cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gt)

    # 5) Adding a spatial reference
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # gdal.RasterizeLayer(ds,bands,layer,burn_values, options = ["BURN_VALUE_FROM=Z"])
    gdal.RasterizeLayer(target_ds, [1], lyrBuffer)
    target_ds = None
    ds.Destroy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", type=str, help="Raster file name  with directory")
    parser.add_argument("--vector", type=str, help="Vector file name  with directory ")
    parser.add_argument("--buffer", type=int, help="Buffer width of line feature")
    parser.add_argument("--output_file", type=str, help="Output image name with directory")
    parser.add_argument("--attribute", type=str, help="Attribute from the vector file")
    args = parser.parse_args()
    rasterize(args)
