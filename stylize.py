#!/usr/bin/env python
import argparse
import logging
from function import adaptive_instance_normalization
import net
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from script_utils.common import common_setup

parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary datasets.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1, help='Number of styles to \
                        create for each image (default: 1)'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'], help='List of image extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )
parser.add_argument('--max-content-size', type=int, default=0,
                    help='Maximum size of content image, \
                    keeping the original size if set to 0'                                                          )
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--seed', default=0, type=int)

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop:
        transform_list.append(torchvision.transforms.CenterCrop(size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def main():
    args = parser.parse_args()

    random.seed(args.seed)

    # set content and style directories
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    style_dir = style_dir.resolve()
    output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    assert style_dir.is_dir(), 'Style directory not found'

    output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, output_dir, args)

    # collect content files
    extensions = args.extensions
    assert len(extensions) > 0, 'No file extensions specified'
    content_dir = Path(content_dir)
    content_dir = content_dir.resolve()
    assert content_dir.is_dir(), 'Content directory not found'
    dataset = []
    for ext in extensions:
        logging.info(f'Finding images with ext: {ext}')
        dataset += list(content_dir.rglob('*.' + ext))

    assert len(dataset) > 0, 'No images with specified extensions found in content directory' + content_dir
    content_paths = sorted(dataset)
    logging.info('Found %d content images in %s' % (len(content_paths), content_dir))

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    logging.info('Found %d style images in %s' % (len(styles), style_dir))

    # Size of the largest image in Kaggle painter by numbers
    Image.MAX_IMAGE_PIXELS = 888150000 + 1

    valid_styles = []
    for s in tqdm(styles, desc='Validating style images'):
        try:
            Image.open(s).convert('RGB')
        except OSError as e:
            logging.warn(f"Skipping style {s} due to error below.")
            continue
        valid_styles.append(s)
    styles = valid_styles

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = input_transform(args.content_size, args.crop)
    style_tf = input_transform(args.style_size, args.crop)

    # Map content path to style path list
    style_paths = {
        content_path: random.sample(styles, args.num_styles)
        for content_path in content_paths
    }

    # actual style transfer as in AdaIN
    with tqdm(total=len(content_paths) * args.num_styles) as pbar:
        for content_path in content_paths:
            logging.debug(f'Processing {content_path}')
            # Construct output dir, check if output already exists.
            rel_path = content_path.relative_to(content_dir)
            out_dir = output_dir.joinpath(rel_path.parent)
            # create directory structure if it does not exist
            if not out_dir.is_dir():
                out_dir.mkdir(parents=True)
            content_name = content_path.stem

            out_paths = [
                out_dir.joinpath(content_name + '-stylized-' +
                                 style_path.stem + content_path.suffix)
                for style_path in style_paths[content_path]
            ]
            if all(x.exists() for x in out_paths):
                logging.debug(
                    f"Outputs for {content_path} already exist at "
                    f"{out_paths}, skipping.")
                pbar.update(len(out_paths))
                continue

            try:
                content_img = Image.open(content_path).convert('RGB')
            except OSError as e:
                logging.info('Skipping stylization of %s due to error below' %(content_path))
                logging.warn(e)
                continue

            if args.max_content_size != 0:
                w, h = content_img.size
                if w > args.max_content_size or h > args.max_content_size:
                    if w > h:
                        ow = args.max_content_size
                        oh = ow * h / w
                    else:
                        oh = args.max_content_size
                        ow = oh * w / h
                    ow, oh = int(ow), int(oh)
                    logging.warn(
                        f"Resizing large image {content_path} with size "
                        f"(w={w}, h={h}) to (ow={ow}, oh={oh})")
                    content_img = content_img.resize((ow, oh), Image.BILINEAR)

            for style_path, output_name in zip(style_paths[content_path],
                                               out_paths):
                if output_name.exists():
                    logging.debug(
                        f"Output {output_name} already exists, skipping")
                    pbar.update(1)
                    continue

                try:
                    style_img = Image.open(style_path).convert('RGB')
                except OSError as e:
                    logging.warn('Skipping stylization of %s with %s due to error below' %(content_path, style_path))
                    logging.warn(e)
                    continue

                content = content_tf(content_img)
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            args.alpha)
                output = output.cpu()

                save_image(output, output_name, padding=0) #default image padding is 2.
                logging.debug(f'Completed {output_name}')
                style_img.close()
                pbar.update(1)
            content_img.close()

if __name__ == '__main__':
    main()
