import argparse
from models import unet_model, resunet_model, segnet_model


def summary(args):
    if args.model == "unet":
        model = unet_model.UNet(args)
        model.summary()
    elif args.model == "resunet":
        model = resunet_model.build_res_unet(args)
        model.summary()
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        model.summary()
    elif args.model == "linknet":
        # pretrained_encoder = 'True',
        # weights_path = './checkpoints/linknet_encoder_weights.h5'
        model = LinkNet(1, input_shape=(256, 256, 3))
        model = model.get_model()
        model.summary()
    elif args.model == "DLinkNet":
        model = segnet_model.create_segnet(args)
        model.summary()
    else:
        print("The model name should be from the unet, resunet, linknet or segnet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    args = parser.parse_args()
    summary(args)
