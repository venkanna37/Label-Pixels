import argparse
from models import lp_utils as lu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--num_classes", type=int, help="Number of classes in label data")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model", default=None)
    args = parser.parse_args()

    model = lu.select_model(args.model, args.weights, tuple(args.input_shape), args.num_classes)
    model.summary()
